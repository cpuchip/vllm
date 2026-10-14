# SPDX-License-Identifier: Apache-2.0
# Scratch instrumentation for syv-ai/HyperQwen#320 (structural tag + spec decode spells </tool_call> in text).
# Never part of the series. Enabled by VLLM_DEBUG_P320=<path prefix>; writes JSON lines to <prefix>.<pid>.jsonl.
# Runner: per step, per structured-output request, every verify row's pre-mask argmax / logit of the added
# </tool_call> token, whether the applied mask allows it, post-mask probabilities, then what was sampled.
# Scheduler: every draft list the grammar trimmed (validate_tokens), with before/after lengths.
import json
import os

import torch

PREFIX = os.environ.get("VLLM_DEBUG_P320", "")
ENABLED = bool(PREFIX)
TC_CLOSE = 248059  # </tool_call> (added token) in the Qwen3.8 tokenizer
LT = 27  # "<"
_step = 0
_fh = None


def _write(rec):
    global _fh
    if _fh is None:
        _fh = open(f"{PREFIX}.{os.getpid()}.jsonl", "a", buffering=1)
    _fh.write(json.dumps(rec) + "\n")


def sched_trim(where, request, proposed, kept):
    if not ENABLED or not request.use_structured_output:
        return
    kept_n = sum(1 for t in kept if t != -1)
    if kept_n != len([t for t in proposed if t != -1]):
        _write({"k": "trim", "where": where, "req": request.request_id,
                "n_out": request.num_output_tokens, "proposed": list(proposed), "kept": list(kept)})


def _rows(scheduler_output, grammar_output, input_batch):
    spec = scheduler_output.scheduled_spec_decode_tokens
    want = set(grammar_output.structured_output_request_ids)
    out, off = [], 0
    for b, req_id in enumerate(input_batch.req_ids):
        drafts = list(spec.get(req_id, ()))
        li = b + off
        off += len(drafts)
        if req_id in want:
            out.append((req_id, b, li, drafts))
    return out


def pre(scheduler_output, grammar_output, input_batch, logits):
    if not ENABLED:
        return None
    reqs = _rows(scheduler_output, grammar_output, input_batch)
    info = []
    for req_id, b, li, drafts in reqs:
        n = 1 + len(drafts)
        x = logits[li:li + n].float()
        top = torch.topk(x, 2, dim=-1)
        info.append({"req": req_id, "b": b, "li": li, "drafts": drafts,
                     "pre_top": top.indices.tolist(), "pre_top_logit": [[round(v, 3) for v in r] for r in top.values.tolist()],
                     "pre_tc": [round(v, 3) for v in x[:, TC_CLOSE].tolist()]})
    return info


def post(info, logits):
    if info is None:
        return
    for r in info:
        n = 1 + len(r["drafts"])
        x = logits[r["li"]:r["li"] + n].float()
        p = x.softmax(dim=-1)
        r["allow_tc"] = torch.isfinite(x[:, TC_CLOSE]).tolist()
        r["allow_lt"] = torch.isfinite(x[:, LT]).tolist()
        r["n_allowed"] = torch.isfinite(x).sum(dim=-1).tolist()
        r["post_top"] = p.argmax(dim=-1).tolist()
        r["p_tc"] = [round(v, 5) for v in p[:, TC_CLOSE].tolist()]
        r["p_lt"] = [round(v, 5) for v in p[:, LT].tolist()]
        d = [t if t >= 0 else 0 for t in r["drafts"]]
        r["p_draft"] = [round(v, 5) for v in p[torch.arange(len(d)), torch.tensor(d, dtype=torch.long)].tolist()] if d else []


def done(info, sampled_token_ids):
    global _step
    _step += 1
    if info is None:
        return
    s = sampled_token_ids.tolist() if torch.is_tensor(sampled_token_ids) else sampled_token_ids
    for r in info:
        r["sampled"] = s[r["b"]] if r["b"] < len(s) else None
        toks = set(r["drafts"]) | set(r["sampled"] or [])
        constrained_tc = any(a and n < 100000 for a, n in zip(r["allow_tc"], r["n_allowed"]))
        interesting = (TC_CLOSE in toks or LT in toks or constrained_tc
                       or any(t[0] == TC_CLOSE for t in r["pre_top"]))
        if interesting:
            r["k"] = "step"
            r["step"] = _step
            _write(r)
