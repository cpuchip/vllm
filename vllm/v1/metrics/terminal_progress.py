# SPDX-License-Identifier: Apache-2.0
"""Request-scoped terminal progress; no prompts, generated text, or GPU syncs."""
import time

from vllm import envs

ENABLED = envs.VLLM_TERMINAL_PROGRESS


def reading(request, completed, *, start=None):
    now = time.monotonic()
    if start is not None and not hasattr(request, "_terminal_started"):
        request._terminal_started = start
        request._terminal_read_last = float("-inf")
    if not hasattr(request, "_terminal_started"):
        return
    if now - request._terminal_read_last < 1.0:
        return
    request._terminal_read_last = now
    print(
        f"[hyperqwen] reading the prompt: {max(0, min(completed, request.num_prompt_tokens)):,} "
        f"of {request.num_prompt_tokens:,} tokens, "
        f"{now - request._terminal_started:.0f} s so far "
        f"(req {request.request_id})", flush=True,
    )


def output(state, timestamp, finish_reason=None, *, cancelled=False):
    n = state.detokenizer.num_output_tokens() if state.detokenizer else 0
    now = time.monotonic()
    ts = timestamp if timestamp is not None else now
    stats = state.stats
    first = getattr(state, "_terminal_first_token", None)
    if n and first is None:
        first = stats.first_token_ts if stats and stats.first_token_ts else ts
        state._terminal_first_token = first
        state._terminal_answer_last = now
        duration = max(0.0, first - stats.scheduled_ts) if stats and stats.scheduled_ts else max(0.0, time.time() - (stats.arrival_time if stats else state._terminal_arrival))
        cached = max(0, state.num_cached_tokens)
        computed = max(0, state.prompt_len - cached)
        rate = computed / duration if duration > 0 else 0.0
        print(
            f"[hyperqwen] prefill: {computed:,} new tokens in {duration:.2f} s "
            f"({rate:.1f} tok/s), cached {cached:,} of {state.prompt_len:,} "
            f"(req {state.external_req_id})", flush=True,
        )
    elapsed = max(0.0, time.time() - (stats.arrival_time if stats else state._terminal_arrival))
    span = max(0.0, ts - first) if first is not None else 0.0
    rate = max(0, n - 1) / span if span > 0 else 0.0
    if finish_reason is not None or cancelled:
        reason = getattr(finish_reason, "name", str(finish_reason)).lower() if not cancelled else "abort"
        print(
            f"[hyperqwen] done: {n:,} tokens in {elapsed:.0f} s ({rate:.1f} tok/s) "
            f"({reason}, cancel={cancelled}) (req {state.external_req_id})", flush=True,
        )
    elif n and now - getattr(state, "_terminal_answer_last", 0.0) >= 1.0:
        state._terminal_answer_last = now
        print(
            f"[hyperqwen] answering: {n:,} of max {state.max_tokens_param} tokens, "
            f"{rate:.1f} tok/s, {elapsed:.0f} s (req {state.external_req_id})", flush=True,
        )
