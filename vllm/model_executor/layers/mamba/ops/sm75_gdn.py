# SPDX-License-Identifier: Apache-2.0
# CUDA recurrent GDN integration for the weicj SM70/SM75 kernel.
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def load_extension():
    from torch.utils.cpp_extension import load
    return load(
        name="vllm_gdn_sm75",
        sources=[str(Path(__file__).with_suffix(".cu"))],
        extra_cuda_cflags=["-O3"],
        extra_cflags=["-O3"],
    )


def chunk_gated_delta_rule(
    q, k, v, g, beta, initial_state, output_final_state,
    cu_seqlens=None, chunk_indices=None, chunk_offsets=None,
    use_qk_l2norm_in_kernel=True, core_attn_out=None,
    aiter_prefill_metadata=None,
):
    from vllm.third_party.flash_linear_attention.ops import (
        chunk_gated_delta_rule as fallback,
    )
    if q.dtype != torch.float16 or q.shape[-1] != 128 or v.shape[-1] != 128:
        return fallback(
            q, k, v, g, beta, initial_state=initial_state,
            output_final_state=output_final_state, cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices, chunk_offsets=chunk_offsets,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            core_attn_out=core_attn_out,
        )
    if use_qk_l2norm_in_kernel:
        from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd
        q, k = l2norm_fwd(q), l2norm_fwd(k)
    # The CUDA port updates its initial state in place. Never expose the live
    # Mamba cache or a caller's float32 state to that write.
    state = initial_state.to(torch.float32, copy=True).contiguous()
    args = (
        q.contiguous(), k.contiguous(), v.contiguous(),
        g.float().contiguous(), beta.float().contiguous(), state,
        q.shape[-1] ** -0.5,
    )
    ext = load_extension()
    if cu_seqlens is None:
        output, final_state = ext.gdn_forward(*args)
    else:
        output, final_state = ext.gdn_forward_varlen(
            *args, cu_seqlens.to(torch.int32).contiguous()
        )
    if core_attn_out is not None:
        core_attn_out.reshape(-1)[:output.numel()].copy_(output.reshape(-1))
    return output, final_state if output_final_state else None
