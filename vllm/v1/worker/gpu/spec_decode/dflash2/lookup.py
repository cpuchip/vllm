"""Lookup-augmented block drafting (LABD).

When the tokens the model just produced already occurred earlier in this request's own
context, the continuation is sitting right there: propose *that* instead of what the model
drafter guessed. One Triton program per request finds the most recent occurrence of the
longest suffix (NMIN..NMAX tokens) of the token history and returns the tokens that
followed it.

Three things make this worth more than a drafter override:

* the proposal is not capped by the drafter's block. The drafter emits the 7 tokens it was
  trained for; a verbatim copy can be dozens of tokens long, and the lookup can fill every
  verify slot the server was launched with (`DFLASH_TOKENS`).
* a match may overlap the suffix it is matched against, so a repeating pattern ("- item"
  every line, an indent, a fence) is proposed from its own period rather than missed.
* the match length is a confidence signal, so the two proposals can be *fused* -- a long
  match takes the whole block, a short one only takes it if the drafter agrees with its
  first tokens. All-or-nothing overriding on a 6-token match loses acceptance on prose.

The proposal stays lossless: the caller rewrites the draft distribution for the overridden
positions to a point mass on the proposed token, which is a legal q for vLLM's rejection
sampler (accept probability becomes p(x), and the residual (p - q)+ is computed from the
same buffer), so the output distribution is unchanged. Greedy requests never read q at all.

Cost is one pass over the request's history (an int32 UVA tensor vLLM already maintains),
with an NMIN-token reject test before any candidate is extended.
"""

import torch

from vllm.triton_utils import tl, triton

# Positions are packed into one int64 score as len * SCORE_STRIDE + end_index, so that
# "longest match, then most recent" is a single max-reduction. max_model_len < SCORE_STRIDE.
SCORE_STRIDE = tl.constexpr(1 << 32)


@triton.jit
def _suffix_lookup_kernel(
    all_token_ids_ptr,
    all_token_ids_stride,
    total_len_ptr,
    idx_mapping_ptr,
    idx_mapping_stride,
    out_tokens_ptr,
    out_tokens_stride,
    out_len_ptr,
    out_valid_ptr,
    search_max,
    k: tl.constexpr,
    NMAX: tl.constexpr,
    NMIN: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_KT: tl.constexpr,
):
    req = tl.program_id(0)
    req_state = tl.load(idx_mapping_ptr + req * idx_mapping_stride)
    if req_state < 0:
        return
    tl.store(out_len_ptr + req, 0)
    tl.store(out_valid_ptr + req, 0)
    total_len = tl.load(total_len_ptr + req_state)
    if total_len < NMIN + 2:
        return
    base = all_token_ids_ptr + req_state.to(tl.int64) * all_token_ids_stride
    end_of_suffix = total_len - 1

    # Candidate match ends run over [lo, hi): any position before the suffix's own end.
    # A candidate may overlap the suffix (a period-p repetition matches at e = t - p),
    # which is how repeated list markers and indentation get proposed.
    hi = end_of_suffix
    lo = tl.maximum(NMIN - 1, total_len - search_max)
    best = tl.full([], -1, tl.int64)
    for start in range(lo, hi, BLOCK):
        e = start + tl.arange(0, BLOCK)
        alive = (e < hi) & (e >= lo)
        # Reject on the NMIN-token prefix first: most blocks die here and never pay for
        # the extension loop.
        for j in range(NMIN):
            s_j = tl.load(base + end_of_suffix - j)
            t_j = tl.load(base + e - j, mask=alive & ((e - j) >= 0), other=-1)
            alive = alive & (t_j == s_j)
        if tl.max(alive.to(tl.int32)) > 0:
            match = tl.where(alive, NMIN, 0)
            longer = alive
            for j in range(NMIN, NMAX):
                s_j = tl.load(
                    base + end_of_suffix - j, mask=(end_of_suffix - j) >= 0, other=-2
                )
                t_j = tl.load(base + e - j, mask=longer & ((e - j) >= 0), other=-1)
                longer = longer & (t_j == s_j)
                match = tl.where(longer, match + 1, match)
            score = tl.where(alive, match.to(tl.int64) * SCORE_STRIDE + e.to(tl.int64), -1)
            best = tl.maximum(best, tl.max(score, axis=0))
    if best < 0:
        return
    match_len = (best // SCORE_STRIDE).to(tl.int32)
    end = (best % SCORE_STRIDE).to(tl.int32)
    # Tokens that actually follow the match inside the history. A match close to the suffix
    # yields fewer than k of them; the caller leaves the rest of the block to the drafter.
    valid = tl.minimum(k, end_of_suffix - end)
    idx = tl.arange(0, BLOCK_KT)
    toks = tl.load(base + end + 1 + idx, mask=idx < valid, other=0)
    tl.store(out_tokens_ptr + req * out_tokens_stride + idx, toks, mask=idx < k)
    tl.store(out_len_ptr + req, match_len)
    tl.store(out_valid_ptr + req, valid)


@triton.jit
def _fuse_draft_kernel(
    draft_tokens_ptr,
    draft_stride,
    lookup_tokens_ptr,
    lookup_stride,
    match_len_ptr,
    valid_ptr,
    use_ptr,
    idx_mapping_ptr,
    idx_mapping_stride,
    hits_ptr,
    take_flags_ptr,
    nmin,
    nstrong,
    agree_min,
    nmin_tail,
    long_min,
    draft_block,
    k: tl.constexpr,
    BLOCK_KT: tl.constexpr,
):
    """Fuse the lookup proposal into the drafted block, and record which positions came
    from the lookup (`use`), which is what the point-mass rewrite consumes.

    The two halves of the block are not the same bet, so they do not share a threshold.

    Head (`idx < draft_block`): the drafter already proposed these, so taking the lookup's
    token *replaces* a considered guess and a coincidental match is a loss. A match of at
    least `nstrong` tokens is taken on its own; a shorter one (down to `nmin`) only if the
    drafter independently proposed the same first `agree_min` tokens -- two independent
    sources agreeing, one having looked at the hidden state and the other at the text.

    Tail (`idx >= draft_block`): nobody proposed these, so anything the match supplies is
    free and the threshold drops to `nmin_tail`. The only requirement is consistency with
    the head that precedes them -- either the lookup owns the head too, or the drafter
    independently agreed with it all the way to `draft_block`. Tail positions always get a
    point mass, since nothing else wrote a draft distribution for them.

    `take_flags` says only that this request has something to put in the tail: a match at
    least `long_min` long with enough tokens left to fill the block. Whether the long block
    is worth its step time is decided by the caller's controller, from what long blocks have
    actually returned (match length is a poor predictor: a 6-token match can carry fifteen
    correct tokens and a 12-token one none)."""
    req = tl.program_id(0)
    # 1 = "no objection": padding rows must not veto a batch-wide decision that is taken
    # only when every active request wants the long block.
    tl.store(take_flags_ptr + req, 1)
    if tl.load(idx_mapping_ptr + req * idx_mapping_stride) < 0:
        return
    tl.store(take_flags_ptr + req, 0)
    idx = tl.arange(0, BLOCK_KT)
    kmask = idx < k
    match_len = tl.load(match_len_ptr + req)
    valid = tl.load(valid_ptr + req)
    drafted = tl.load(draft_tokens_ptr + req * draft_stride + idx, mask=kmask, other=0)
    # draft_tokens is int64 (vllm/v1/worker/gpu/spec_decode/speculator.py); the history the
    # lookup reads is int32, so the proposal is widened before it is written back.
    looked = tl.load(lookup_tokens_ptr + req * lookup_stride + idx, mask=kmask, other=0)
    looked = looked.to(tl.int64)
    # Leading agreement between the two proposals, over the drafter's own block.
    disagree = (drafted != looked) & (idx < valid) & (idx < draft_block) & kmask
    # Positions the match did not supply are not agreement: without this a match with one
    # or two tokens left scores a full-block agreement and walks through the gate.
    agree = tl.minimum(tl.min(tl.where(disagree, idx, draft_block)), valid)
    take_head = (match_len >= nstrong) | ((match_len >= nmin) & (agree >= agree_min))
    tail = idx >= draft_block
    take_tail = (match_len >= nmin_tail) & (take_head | (agree >= draft_block))
    from_lookup = tl.where(tail, take_tail, take_head) & (idx < valid) & kmask
    tl.store(draft_tokens_ptr + req * draft_stride + idx, looked, mask=from_lookup)
    use = (from_lookup | tail) & kmask
    tl.store(use_ptr + req * k + idx, use.to(tl.int32), mask=kmask)
    # Asking for the long block costs step time on every request in the batch whether or
    # not the tail is accepted, so it takes evidence that a copy is *running*: the step that
    # just finished accepted every token it was given (prev_acc = 1 + accepted, and a short
    # block can produce at most 1 + draft_block), and the lookup has a long enough match
    # with enough tokens left to fill the tail. Match length alone is a poor predictor --
    # measured, it needs a threshold of 22 before it stops losing on prose, by which point
    # it has given up most of the win on copy work.
    # "This request has something to put in the tail." Whether the long block is worth its
    # step time is decided by the caller: this kernel runs inside the draft model's captured
    # CUDA graph, where host-side state is frozen at capture time.
    has_tail = (match_len >= long_min) & (valid > draft_block) & take_tail
    tl.store(take_flags_ptr + req, has_tail.to(tl.int32))
    tl.atomic_add(hits_ptr, take_head.to(tl.int64))


@triton.jit
def _point_mass_draft_logits_kernel(
    draft_logits_ptr,
    cached_candidate_ptr,
    draft_tokens_ptr,
    draft_tokens_stride,
    use_ptr,
    idx_mapping_ptr,
    idx_mapping_stride,
    draft_logits_stride_0,
    draft_logits_stride_1,
    num_steps: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Rewrite one (request, step) row of the sparse draft-logit cache to a point mass on the
    proposed token, keeping the cache's invariant that every finite entry of a row is listed
    in `cached_candidate_ids` (so the next step erases it). The token is read back from
    draft_tokens, so q always describes the proposal that is actually verified."""
    flat = tl.program_id(0)
    req = flat // num_steps
    step = flat % num_steps
    req_state = tl.load(idx_mapping_ptr + req * idx_mapping_stride)
    if req_state < 0:
        return
    if tl.load(use_ptr + req * num_steps + step) == 0:
        return
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < top_k
    cache_base = (req_state * num_steps + step) * top_k
    logits_base = (
        draft_logits_ptr + req_state * draft_logits_stride_0 + step * draft_logits_stride_1
    )
    old = tl.load(cached_candidate_ptr + cache_base + offsets, mask=mask)
    tl.store(logits_base + old, -float("inf"), mask=mask)
    token = tl.load(draft_tokens_ptr + req * draft_tokens_stride + step)
    tl.store(logits_base + token, 0.0)
    tl.store(cached_candidate_ptr + cache_base + offsets, token, mask=mask)


def suffix_lookup(
    all_token_ids: torch.Tensor,
    total_len: torch.Tensor,
    idx_mapping: torch.Tensor,
    num_reqs: int,
    k: int,
    idx_mapping_stride: int = 1,
    nmax: int = 32,
    nmin: int = 4,
    search_max: int = 1 << 30,
    out_tokens: torch.Tensor | None = None,
    out_len: torch.Tensor | None = None,
    out_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (tokens [num_reqs, k] int32, match_len [num_reqs] int32; 0 = no match,
    valid [num_reqs] int32 = how many of the k tokens the match actually supplied).

    idx_mapping maps batch position -> request-state index, read with idx_mapping_stride
    (the DFlash speculators keep it as sample_idx_mapping, one entry per (request, step))."""
    device = idx_mapping.device
    if out_tokens is None:
        out_tokens = torch.zeros((num_reqs, k), dtype=torch.int32, device=device)
    if out_len is None:
        out_len = torch.zeros(num_reqs, dtype=torch.int32, device=device)
    if out_valid is None:
        out_valid = torch.zeros(num_reqs, dtype=torch.int32, device=device)
    _suffix_lookup_kernel[(num_reqs,)](
        all_token_ids,
        all_token_ids.stride(0),
        total_len,
        idx_mapping,
        idx_mapping_stride,
        out_tokens,
        out_tokens.stride(0),
        out_len,
        out_valid,
        search_max,
        k=k,
        NMAX=nmax,
        NMIN=nmin,
        BLOCK=1024,
        BLOCK_KT=triton.next_power_of_2(k),
        num_warps=4,
    )
    return out_tokens, out_len, out_valid


def fuse_draft(
    draft_tokens: torch.Tensor,
    lookup_tokens: torch.Tensor,
    match_len: torch.Tensor,
    valid: torch.Tensor,
    use: torch.Tensor,
    idx_mapping: torch.Tensor,
    hits: torch.Tensor,
    num_reqs: int,
    k: int,
    draft_block: int | None = None,
    idx_mapping_stride: int = 1,
    nmin: int = 6,
    nstrong: int = 8,
    agree_min: int = 2,
    nmin_tail: int = 4,
    long_min: int = 6,
    take_flags: torch.Tensor | None = None,
) -> None:
    """In-place: draft_tokens[:num_reqs] <- fused proposal, use <- overridden positions.

    draft_block is how many of the k verify positions the drafter itself proposed; the
    rest are the lookup's to fill (defaults to all k, i.e. the drafter filled the block)."""
    assert draft_tokens.dtype == torch.int64
    if take_flags is None:
        take_flags = torch.zeros(num_reqs, dtype=torch.int32, device=draft_tokens.device)
    _fuse_draft_kernel[(num_reqs,)](
        draft_tokens,
        draft_tokens.stride(0),
        lookup_tokens,
        lookup_tokens.stride(0),
        match_len,
        valid,
        use,
        idx_mapping,
        idx_mapping_stride,
        hits,
        take_flags,
        nmin,
        nstrong,
        agree_min,
        nmin_tail,
        long_min,
        k if draft_block is None else draft_block,
        k=k,
        BLOCK_KT=triton.next_power_of_2(k),
        num_warps=1,
    )
