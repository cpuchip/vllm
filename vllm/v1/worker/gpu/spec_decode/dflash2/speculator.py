# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Any

import torch

import vllm.envs as envs
from vllm.config.compilation import CUDAGraphMode
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

logger = init_logger(__name__)
from vllm.v1.worker.gpu.spec_decode.dflash2.lookup import (
    _point_mass_draft_logits_kernel,
    fuse_draft,
    suffix_lookup,
)


@triton.jit
def _selector_walk_kernel(
    scores_ptr,
    candidate_ptr,
    sample_pos_ptr,
    req_state_ptr,
    temperature_ptr,
    seeds_ptr,
    tokens_ptr,
    realized_scores_ptr,
    req_top_p_ptr,
    req_top_k_ptr,
    num_steps: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SAMPLE_PROBABILISTIC: tl.constexpr,
    USE_FP64: tl.constexpr,
    TRUNCATE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < top_k
    req_state = tl.load(req_state_ptr + row * num_steps)
    valid = req_state >= 0
    temperature = tl.load(temperature_ptr + req_state, mask=valid, other=0.0)
    seed = tl.load(seeds_ptr + req_state, mask=valid, other=0)
    if TRUNCATE:
        req_top_p = tl.load(req_top_p_ptr + req_state, mask=valid, other=1.0)
        req_top_k = tl.load(req_top_k_ptr + req_state, mask=valid, other=BLOCK_K)
    previous = 0
    for step in range(num_steps):
        flat = row * num_steps + step
        score_base = (flat * top_k + previous) * top_k
        scores = tl.load(
            scores_ptr + score_base + offsets,
            mask=mask & valid,
            other=float("-inf"),
        ).to(tl.float32)
        candidate_base = flat * top_k
        candidates = tl.load(
            candidate_ptr + candidate_base + offsets,
            mask=mask & valid,
            other=0,
        )

        if SAMPLE_PROBABILISTIC and temperature != 0.0 and TRUNCATE:
            # Apply the request's top-k/top-p to the proposal distribution over the
            # candidates. The rejection sampler consumes pre-temperature logits, so keep
            # the original scores and use -inf to represent the truncated support.
            scaled_scores = scores / temperature
            mx = tl.max(scaled_scores, axis=0)
            probs = tl.exp(scaled_scores - mx)
            probs = probs / tl.sum(probs, axis=0)
            greater = scaled_scores[None, :] > scaled_scores[:, None]
            rank = tl.sum(greater.to(tl.int32), axis=1)
            mass_before = tl.sum(tl.where(greater, probs[None, :], 0.0), axis=1)
            keep = mask & (rank < req_top_k) & (mass_before < req_top_p)
            scores = tl.where(keep, scores, -float("inf"))

        # sample_pos is the predicted token's position P. Sampling keys a draw
        # by the position before the sampled token, P-1.
        sample_pos = tl.load(sample_pos_ptr + flat) - 1
        _, index = gumbel_noised_argmax(
            scores,
            candidates,
            mask & valid,
            seed,
            sample_pos,
            temperature if SAMPLE_PROBABILISTIC else 0.0,
            IS_DRAFTING=True,
            USE_FP64=USE_FP64,
        )
        # vLLM 0.28.0's rejection sampler expects pre-temperature logits. With TRUNCATE
        # the cached row is the truncated proposal (-inf outside the kept support).
        realized = scores
        tl.store(
            realized_scores_ptr + candidate_base + offsets,
            realized,
            mask=mask & valid,
        )
        token = tl.load(candidate_ptr + candidate_base + index, mask=valid, other=0)
        tl.store(tokens_ptr + flat, token, mask=valid)
        previous = index


@triton.jit
def _cache_draft_logits_kernel(
    draft_logits_ptr,
    cached_candidate_ptr,
    candidate_ptr,
    scores_ptr,
    req_state_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    num_steps: tl.constexpr,
    cache_steps: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    flat = tl.program_id(0)
    req_state = tl.load(req_state_ptr + flat)
    step = flat % num_steps
    offsets = tl.arange(0, BLOCK_K)
    mask = (req_state >= 0) & (offsets < top_k)
    candidate_base = flat * top_k
    # The candidate cache spans the whole verify block, of which the drafter fills the
    # first num_steps rows (dflash2/lookup.py fills the rest).
    cache_base = (req_state * cache_steps + step) * top_k
    old_token_ids = tl.load(cached_candidate_ptr + cache_base + offsets, mask=mask)
    logits_base = (
        draft_logits_ptr
        + req_state * draft_logits_stride_0
        + step * draft_logits_stride_1
    )
    tl.store(logits_base + old_token_ids, -float("inf"), mask=mask)
    token_ids = tl.load(candidate_ptr + candidate_base + offsets, mask=mask)
    scores = tl.load(scores_ptr + candidate_base + offsets, mask=mask)
    tl.store(logits_base + token_ids, scores, mask=mask)
    tl.store(cached_candidate_ptr + cache_base + offsets, token_ids, mask=mask)


class DFlash2Speculator(DFlashSpeculator):
    """DFlash2 = DFlash + local convolutions + candidate selector, optionally with
    lookup-augmented drafting (VLLM_DFLASH2_LOOKUP=1)."""

    _speculator_name = "DFlash2"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        draft_config = self.draft_model_config.hf_config.dflash_config
        self.selector_top_k = int(draft_config["selector_top_k"])
        self._anchor_indices = (
            torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
            * self.num_query_per_req
        )
        self._selector_tokens = torch.empty(
            (self.max_num_reqs, self.draft_block),
            dtype=self.draft_tokens.dtype,
            device=device,
        )
        self._selector_scores = torch.empty(
            self.max_num_reqs,
            self.draft_block,
            self.selector_top_k,
            dtype=torch.float32,
            device=device,
        )
        self._cached_candidate_ids = torch.zeros(
            (self.max_num_reqs, self.num_speculative_steps, self.selector_top_k),
            dtype=torch.int64,
            device=device,
        )
        # Request top_p/top_k buffers ([max_num_reqs], indexed by request state), handed
        # over by the model runner; VLLM_DFLASH2_DRAFT_TOPK_TOPP=0 disables the truncation.
        self._req_top_p: torch.Tensor | None = None
        self._req_top_k: torch.Tensor | None = None
        self._truncate = envs.VLLM_DFLASH2_DRAFT_TOPK_TOPP
        # Lookup-augmented block drafting: propose the continuation of an earlier
        # occurrence of the current suffix instead of the model's guess (lookup.py).
        self._lookup = envs.VLLM_DFLASH2_LOOKUP
        # Three separate questions, three thresholds (dflash2/lookup.py):
        #  NMIN/NSTRONG/AGREE   when to let the lookup replace a token the drafter proposed
        #  NMIN_TAIL            when to fill the positions the drafter never proposed
        #  LONGMIN              when the long verify block is worth its step time
        self._lookup_nmin = envs.VLLM_DFLASH2_LOOKUP_NMIN
        # 12, not 32: the kernel picks the longest match and breaks ties by recency, and a
        # longer cap makes it prefer an older long match over a newer short one. On
        # quote-and-explain work the newer one is the better predictor (3.21 vs 2.69 tokens
        # per step), and copies match well past 12 either way.
        self._lookup_nmax = envs.VLLM_DFLASH2_LOOKUP_NMAX
        # NSTRONG = NMIN and AGREE = 0 means "take any match of NMIN or more" for the
        # positions the drafter also proposed. Measured better at C1 (3.33 vs 3.27 tokens
        # per step) than requiring drafter agreement for medium matches.
        self._lookup_nstrong = envs.VLLM_DFLASH2_LOOKUP_NSTRONG
        self._lookup_agree = envs.VLLM_DFLASH2_LOOKUP_AGREE
        self._lookup_nmin_tail = envs.VLLM_DFLASH2_LOOKUP_NMIN_TAIL
        self._lookup_long_min = envs.VLLM_DFLASH2_LOOKUP_LONGMIN
        # Below this many tokens of context the long block is taken unconditionally, on the
        # theory that an extra verify position is nearly free there. Off by default: it
        # measured +8% at C1 under one memory configuration and -13% under the one that
        # ships (4 request slots, 56k), because what the extra positions cost depends on the
        # paged-attention layout as much as on the KV length.
        self._lookup_cheap_ctx = envs.VLLM_DFLASH2_LOOKUP_CHEAP_CTX
        self._lookup_search = envs.VLLM_DFLASH2_LOOKUP_SEARCH
        self._req_states = None
        self._lookup_tokens = torch.zeros(
            (self.max_num_reqs, self.num_speculative_steps), dtype=torch.int32, device=device
        )
        self._lookup_len = torch.zeros(self.max_num_reqs, dtype=torch.int32, device=device)
        self._lookup_valid = torch.zeros(self.max_num_reqs, dtype=torch.int32, device=device)
        # Which draft positions came from the lookup, per request: what the point-mass
        # rewrite of the draft distribution applies to.
        self._lookup_use = torch.zeros(
            (self.max_num_reqs, self.num_speculative_steps), dtype=torch.int32, device=device
        )
        self._lookup_hits = torch.zeros((), dtype=torch.int64, device=device)
        # Adaptive verify length: a long block only pays for itself while the request is
        # reproducing its context, so the scheduler is asked for the drafter's own block
        # otherwise. The signal is the previous step's lookup decision, read from a pinned
        # copy -- one step stale by construction, which costs at most one step at each end
        # of a copy run and never needs a synchronise.
        self._adaptive = (
            envs.VLLM_DFLASH2_LOOKUP_ADAPTIVE
            and self.draft_block < self.num_speculative_steps
        )
        self._take_flags = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        self._last_num_reqs = 0
        self._flags_cpu = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device="cpu"
        ).pin_memory()
        self._flags_n = 0
        self._prev_want = False
        # Entering the long block takes two qualifying steps in a row and leaving it took
        # one, so a single step where the flag dropped out mid-copy cost three. It drops out
        # for reasons that have nothing to do with the copy ending -- a line the lookup
        # cannot match, or a flag copy that had not landed yet -- and the same server then
        # measured 13.76, 13.92, 13.92, 13.92 and 13.92 tokens per step on consecutive runs
        # of one prompt against 14.97 on the first. STICKY holds the long block for that
        # many steps after the flag goes out, which makes it 15.21 every time.
        #
        # Gating the hold on "the request is still emitting a full block a step" -- which
        # should distinguish a late flag from a finished copy -- removes the whole effect
        # (13.92 again): by the time the flag drops the step it describes was not saturated
        # either, so the two are not independent evidence.
        #
        # It only applies with one request in flight, and that restriction is not caution.
        # This counter is one number for the whole batch, and unlike the entry condition it
        # keeps the long block on through steps where the flags say no -- so with several
        # requests, which block length a copying request gets starts to depend on when the
        # others arrived. Different block length, different rounding, different greedy text:
        # bench/labd_soak.py caught a verbatim copy coming out differently in two rounds of
        # an otherwise identical 4-way batch, and reproducibly did not with STICKY=0.
        self._lookup_sticky = envs.VLLM_DFLASH2_LOOKUP_STICKY
        self._sticky = 0

        if self._lookup:
            logger.info(
                "DFlash2 lookup-augmented drafting on (k=%d nmin=%d nmax=%d nstrong=%d "
                "agree=%d nmin_tail=%d longmin=%d search=%d)",
                self.num_speculative_steps, self._lookup_nmin, self._lookup_nmax,
                self._lookup_nstrong, self._lookup_agree, self._lookup_nmin_tail,
                self._lookup_long_min, self._lookup_search,
            )

    def next_num_draft_tokens(self) -> int | None:
        """How many of the proposed tokens the scheduler should put up for verification next
        step (None = all of them).

        This runs on the host once per step, which is why the decision lives here and not in
        `_apply_lookup`: the draft pass is replayed from a captured CUDA graph, so anything
        Python does in there runs at capture time only. The two inputs are the per-request
        flag the (replayed) fuse kernel writes -- "there is something to put in the tail" --
        and what the step that just finished emitted. A long block costs step time on every
        request in the batch whether or not its tail is accepted, so it takes both, and
        unanimity across the batch. Below CHEAP_CTX tokens of context an extra verify
        position is nearly free (+6% per step at 1.5k against +27% at 25k), so there the
        flag alone is enough."""
        if not self._adaptive:
            return None
        emitted = getattr(self, "last_num_emitted", None)
        if emitted is None:
            return self.draft_block
        if self.draft_max_seq_len <= self._lookup_cheap_ctx:
            # An extra verify position costs attention proportional to the KV length: +6%
            # per step at 1.5k of context against +27% at 25k. Below the threshold it is
            # cheap enough that taking the long block unconditionally wins (+8% at C1),
            # even on the steps where the lookup has nothing to put in it.
            return self.num_speculative_steps
        num_reqs = emitted.shape[0]
        # Read the decision that landed from the previous step and start the copy for the
        # next one. Reading it synchronously (`.item()`) is a device synchronise on every
        # decode step and measured 5% -- more than the long block itself is worth on most
        # work. One step of staleness costs a short step at the start of a copy run and a
        # long one at its end.
        want = bool(self._flags_n and self._flags_cpu[: self._flags_n].all())
        fused = (self._take_flags[:num_reqs] > 0) & (emitted >= 1 + self.draft_block)
        self._flags_cpu[:num_reqs].copy_(fused, non_blocking=True)
        self._flags_n = num_reqs
        # Two qualifying steps in a row, not one: a single saturated step happens in the
        # middle of ordinary prose (a quoted phrase, a repeated list marker) and the long
        # block it buys is then wasted. Waiting for the second one costs the first step of
        # a copy and removes the loss on quote-and-explain work.
        if want and self._prev_want:
            long_block = True
            self._sticky = self._lookup_sticky if num_reqs == 1 else 0
        elif self._sticky > 0 and num_reqs == 1:
            # Coasting: re-entry costs two steps, so a run that has already earned the long
            # block twice gets the benefit of the doubt for a few more.
            long_block, self._sticky = True, self._sticky - 1
        else:
            long_block, self._sticky = False, 0
        self._prev_want = want
        return self.num_speculative_steps if long_block else self.draft_block

    def set_req_states(self, req_states) -> None:
        """The token history the lookup matches against (set by the model runner)."""
        self._req_states = req_states

    def draft_logits_spec(self, vllm_config: VllmConfig) -> tuple[torch.dtype, float]:
        # The v0.28 rejection sampler divides cached logits by temperature itself.
        # -inf is required because the sparse cache kernel only writes the K candidates.
        return torch.float32, -float("inf")

    def set_sampling_states(self, sampling_states) -> None:
        self._req_top_p = sampling_states.top_p.gpu
        self._req_top_k = sampling_states.top_k.gpu

    def _sample_path(
        self,
        candidate_ids: torch.Tensor,
        scores: torch.Tensor,
        num_reqs: int,
    ) -> None:
        block_k = triton.next_power_of_2(self.selector_top_k)
        truncate = self._truncate and self._req_top_p is not None
        _selector_walk_kernel[(num_reqs,)](
            scores.contiguous(),
            candidate_ids.contiguous(),
            self.sample_pos,
            self.sample_idx_mapping,
            self.temperature,
            self.seeds,
            self._selector_tokens,
            self._selector_scores,
            self._req_top_p if truncate else self.temperature,
            self._req_top_k if truncate else self.sample_idx_mapping,
            num_steps=self.draft_block,
            top_k=self.selector_top_k,
            BLOCK_K=block_k,
            SAMPLE_PROBABILISTIC=self.draft_logits is not None,
            USE_FP64=self.use_fp64_gumbel,
            TRUNCATE=truncate,
            num_warps=1,
        )

    def _cache_draft_logits(self, candidate_ids: torch.Tensor, num_sample: int) -> None:
        draft_logits = self.draft_logits
        if draft_logits is None:
            # Greedy rejection sampling never reads q; the token fusion above is still
            # useful because it supplies the proposal handed to the target.
            return
        block_k = triton.next_power_of_2(self.selector_top_k)
        _cache_draft_logits_kernel[(num_sample,)](
            draft_logits,
            self._cached_candidate_ids,
            candidate_ids,
            self._selector_scores,
            self.sample_idx_mapping,
            draft_logits.stride(0),
            draft_logits.stride(1),
            num_steps=self.draft_block,
            cache_steps=self.num_speculative_steps,
            top_k=self.selector_top_k,
            BLOCK_K=block_k,
            num_warps=1,
        )

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        last_hidden_states = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        num_sample = num_reqs * self.draft_block
        hidden_states = last_hidden_states[self.sample_indices[:num_sample]].view(
            num_reqs, self.draft_block, -1
        )
        candidate_ids, unary_logits = self.model.compute_candidates(
            hidden_states.flatten(0, 1)
        )
        candidate_ids = candidate_ids.view(
            num_reqs, self.draft_block, self.selector_top_k
        )
        unary_logits = unary_logits.view_as(candidate_ids)
        anchor_token_ids = self.input_buffers.input_ids[self._anchor_indices[:num_reqs]]
        scores = self.model.model.candidate_selector(
            candidate_ids,
            unary_logits,
            hidden_states,
            anchor_token_ids,
        )
        self._sample_path(candidate_ids, scores, num_reqs)
        if self.enable_adaptive_verification:
            self._maybe_predict_acceptance(
                self._selector_scores[:num_reqs].flatten(0, 1),
                self.sample_idx_mapping[:num_sample],
                self.sample_col[:num_sample],
            )
        if self.draft_logits is not None:
            self._cache_draft_logits(candidate_ids, num_sample)
        self.draft_tokens[:num_reqs, : self.draft_block].copy_(
            self._selector_tokens[:num_reqs]
        )
        self._apply_lookup(num_reqs)

    def _apply_lookup(self, num_reqs: int) -> None:
        """Fuse the drafted block with the continuation of an earlier occurrence of the
        current suffix, where one exists (see dflash2/lookup.py), and fill the verify
        positions past the drafter's own block. The draft distribution for every position
        the lookup supplied becomes a point mass on the proposed token, which keeps vLLM's
        rejection sampling exact."""
        if not self._lookup or self._req_states is None:
            return
        tokens, match_len, valid = suffix_lookup(
            self._req_states.all_token_ids.gpu,
            self._req_states.total_len.gpu,
            self.sample_idx_mapping,
            num_reqs,
            self.num_speculative_steps,
            idx_mapping_stride=self.draft_block,
            nmax=self._lookup_nmax,
            nmin=self._lookup_nmin,
            search_max=self._lookup_search,
            out_tokens=self._lookup_tokens[:num_reqs],
            out_len=self._lookup_len[:num_reqs],
            out_valid=self._lookup_valid[:num_reqs],
        )
        fuse_draft(
            self.draft_tokens[:num_reqs],
            tokens,
            match_len,
            valid,
            self._lookup_use[:num_reqs],
            self.sample_idx_mapping,
            self._lookup_hits,
            num_reqs,
            self.num_speculative_steps,
            draft_block=self.draft_block,
            idx_mapping_stride=self.draft_block,
            nmin=self._lookup_nmin,
            nstrong=self._lookup_nstrong,
            agree_min=self._lookup_agree,
            nmin_tail=self._lookup_nmin_tail,
            long_min=self._lookup_long_min,
            take_flags=self._take_flags[:num_reqs],
        )
        draft_logits = self.draft_logits
        if draft_logits is None:
            return
        block_k = triton.next_power_of_2(self.selector_top_k)
        _point_mass_draft_logits_kernel[(num_reqs * self.num_speculative_steps,)](
            draft_logits,
            self._cached_candidate_ids,
            self.draft_tokens,
            self.draft_tokens.stride(0),
            self._lookup_use,
            self.sample_idx_mapping,
            self.draft_block,
            draft_logits.stride(0),
            draft_logits.stride(1),
            num_steps=self.num_speculative_steps,
            top_k=self.selector_top_k,
            BLOCK_K=block_k,
            num_warps=1,
        )
