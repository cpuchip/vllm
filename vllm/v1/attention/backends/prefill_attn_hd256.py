"""Prefill attention for head_dim 256 on sm86: GQA-packed, paged-KV, causal,
with an int8-QK^T mode (SageAttention-style) that runs the score GEMM on int8
tensor cores at 2x the fp16 rate.

Geometry (Qwen3.8-27B full-attention layers): 24 q-heads / 4 kv-heads (G=6),
D=256, bf16 paged KV [num_blocks, block_size, Hkv, D], chunked prefill
(q_len<=2048 new tokens against kv_len total context), causal.

Why int8 QK is exact enough: K is smoothed by its per-(head, d) mean over the
context before quantization. For a fixed query row, q . mean_k is the same for
every key, so the softmax is invariant to the subtraction — the correction
term is never computed. Q rows and K tiles carry their own absmax scales.
P.V stays bf16 (fp32 accumulate).

Grid is (q-tiles, q-heads) like FA2 — the six sibling q-heads of a kv head
run concurrently, so their K/V tile reads mostly hit L2; the kernel is
compute-bound on this card either way (FA2 measures 85% of the fp16 mma
ceiling), which is exactly why the int8 mode exists.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _k_stats_kernel(
    k_ptr, bt_ptr, mean_ptr, seqused_ptr,
    stride_kb, stride_ks, stride_kh, stride_bt,
    BLOCK_SIZE: tl.constexpr, D: tl.constexpr, CHUNK: tl.constexpr,
):
    # per (kv-head): mean of K over the whole context, fp32 [Hkv, D]
    kvh = tl.program_id(0)
    kv_len = tl.load(seqused_ptr)
    d = tl.arange(0, D)
    acc = tl.zeros([D], dtype=tl.float32)
    for start in range(0, kv_len, CHUNK):
        pos = start + tl.arange(0, CHUNK)
        ok = pos < kv_len
        blk = tl.load(bt_ptr + pos // BLOCK_SIZE, mask=ok, other=0)
        slot = pos % BLOCK_SIZE
        ptrs = k_ptr + blk[:, None] * stride_kb + slot[:, None] * stride_ks \
            + kvh * stride_kh + d[None, :]
        k = tl.load(ptrs, mask=ok[:, None], other=0.0).to(tl.float32)
        acc += tl.sum(k, axis=0)
    tl.store(mean_ptr + kvh * D + d, acc / kv_len)


@triton.jit
def _k_quant_kernel(
    k_ptr, bt_ptr, kmean_ptr, k8_ptr, kscale_ptr, seqused_ptr,
    stride_kb, stride_ks, stride_kh,
    stride_k8n, stride_k8h,
    HKV: tl.constexpr, BLOCK_SIZE: tl.constexpr, D: tl.constexpr,
    TILE: tl.constexpr,
):
    """Gather paged K, subtract the head mean, absmax-quantize per
    (TILE keys, head) into contiguous int8 scratch + fp32 scales."""
    pid = tl.program_id(0)
    kvh = tl.program_id(1)
    kv_len = tl.load(seqused_ptr)
    pos = pid * TILE + tl.arange(0, TILE)
    ok = pos < kv_len
    d = tl.arange(0, D)
    blk = tl.load(bt_ptr + pos // BLOCK_SIZE, mask=ok, other=0)
    slot = pos % BLOCK_SIZE
    ptrs = k_ptr + blk[:, None] * stride_kb + slot[:, None] * stride_ks \
        + kvh * stride_kh + d[None, :]
    mean = tl.load(kmean_ptr + kvh * D + d)
    k = tl.load(ptrs, mask=ok[:, None], other=0.0).to(tl.float32) - mean[None, :]
    k = tl.where(ok[:, None], k, 0.0)
    amax = tl.maximum(tl.max(tl.abs(k)), 1e-8)
    k8 = (k * (127.0 / amax) + 0.5 * tl.where(k >= 0, 1.0, -1.0)).to(tl.int8)
    o_ptrs = k8_ptr + pos[:, None] * stride_k8n + kvh * stride_k8h + d[None, :]
    tl.store(o_ptrs, k8, mask=ok[:, None])
    tl.store(kscale_ptr + pid * HKV + kvh, amax / 127.0)


@triton.jit
def _prefill_attn_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, bt_ptr, k8_ptr, kscale_ptr,
    seqused_ptr, q_start_pos,
    sm_scale,
    stride_qt, stride_qh,
    stride_kb, stride_ks, stride_kh,
    stride_vb, stride_vs, stride_vh,
    stride_k8n, stride_k8h,
    stride_ot, stride_oh,
    stride_bt,
    q_len,
    HKV: tl.constexpr, KTILE: tl.constexpr,
    G: tl.constexpr, D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_N: tl.constexpr,
    INT8_QK: tl.constexpr,
):
    pid_t = tl.program_id(0)          # q-token tile
    qh = tl.program_id(1)             # q head
    kvh = qh // G                     # its kv head
    kv_len = tl.load(seqused_ptr)

    M: tl.constexpr = BLOCK_T         # rows = tokens (power of 2)
    tok = pid_t * BLOCK_T + tl.arange(0, M)     # local q token index
    row_ok = tok < q_len
    d = tl.arange(0, D)

    # global position of each q row (for causal masking against absolute kv pos)
    q_pos = q_start_pos + tok

    q_ptrs = q_ptr + tok[:, None] * stride_qt + qh * stride_qh + d[None, :]
    q = tl.load(q_ptrs, mask=row_ok[:, None], other=0.0)

    if INT8_QK:
        # per-row absmax int8 quantization of q (K is pre-quantized)
        q_f = q.to(tl.float32)
        q_amax = tl.maximum(tl.max(tl.abs(q_f), axis=1), 1e-8)
        q_i8 = (q_f * (127.0 / q_amax)[:, None] + 0.5 * tl.where(q_f >= 0, 1.0, -1.0))
        q_i8 = q_i8.to(tl.int8)
        q_scale = q_amax / 127.0                 # [M]

    m_i = tl.full([M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([M], dtype=tl.float32)
    DH: tl.constexpr = D // 2
    acc0 = tl.zeros([M, DH], dtype=tl.float32)
    acc1 = tl.zeros([M, DH], dtype=tl.float32)

    # the last kv position any row in this tile may see
    hi = q_start_pos + tl.minimum(pid_t * BLOCK_T + BLOCK_T, q_len)
    for start in range(0, hi, BLOCK_N):
        pos = start + tl.arange(0, BLOCK_N)
        k_ok = pos < kv_len
        blk = tl.load(bt_ptr + pos // BLOCK_SIZE, mask=k_ok, other=0)
        slot = pos % BLOCK_SIZE

        if INT8_QK:
            k8_ptrs = k8_ptr + pos[:, None] * stride_k8n + kvh * stride_k8h + d[None, :]
            k_i8 = tl.load(k8_ptrs, mask=k_ok[:, None], other=0)   # [N, D] int8
            k_sc = tl.load(kscale_ptr + (start // KTILE) * HKV + kvh)
            s_i32 = tl.dot(q_i8, tl.trans(k_i8), out_dtype=tl.int32)
            s = s_i32.to(tl.float32) * (q_scale[:, None] * (k_sc * sm_scale))
        else:
            k_ptrs = k_ptr + blk[:, None] * stride_kb + slot[:, None] * stride_ks \
                + kvh * stride_kh + d[None, :]
            k = tl.load(k_ptrs, mask=k_ok[:, None], other=0.0)     # [N, D] bf16
            s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale

        # causal + validity mask on absolute positions
        causal_ok = (pos[None, :] <= q_pos[:, None]) & k_ok[None, :] & row_ok[:, None]
        s = tl.where(causal_ok, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        # guard rows that have seen nothing yet
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        alpha = tl.where(m_i == float("-inf"), 0.0, alpha)
        p = tl.exp(s - m_safe[:, None])
        p = tl.where(causal_ok, p, 0.0)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc0 = acc0 * alpha[:, None]
        acc1 = acc1 * alpha[:, None]

        p_b = p.to(tl.bfloat16)
        v_ptrs0 = v_ptr + blk[:, None] * stride_vb + slot[:, None] * stride_vs \
            + kvh * stride_vh + tl.arange(0, DH)[None, :]
        v0 = tl.load(v_ptrs0, mask=k_ok[:, None], other=0.0)
        acc0 += tl.dot(p_b, v0, out_dtype=tl.float32)
        v_ptrs1 = v_ptr + blk[:, None] * stride_vb + slot[:, None] * stride_vs \
            + kvh * stride_vh + (DH + tl.arange(0, DH))[None, :]
        v1 = tl.load(v_ptrs1, mask=k_ok[:, None], other=0.0)
        acc1 += tl.dot(p_b, v1, out_dtype=tl.float32)
        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    o0 = acc0 / l_safe[:, None]
    o1 = acc1 / l_safe[:, None]
    o_ptrs0 = o_ptr + tok[:, None] * stride_ot + qh * stride_oh \
        + tl.arange(0, DH)[None, :]
    tl.store(o_ptrs0, o0.to(tl.bfloat16), mask=row_ok[:, None])
    o_ptrs1 = o_ptr + tok[:, None] * stride_ot + qh * stride_oh \
        + (DH + tl.arange(0, DH))[None, :]
    tl.store(o_ptrs1, o1.to(tl.bfloat16), mask=row_ok[:, None])


def prefill_attn(q, k_cache, v_cache, out, block_table, kv_len, q_start_pos,
                 sm_scale, int8_qk=True, block_t=16, block_n=64,
                 num_warps=8, num_stages=2, kmean=None):
    """Single-request chunked-prefill attention.

    q, out: [q_len, Hq, D] bf16; k_cache/v_cache: [blocks, bs, Hkv, D] bf16;
    block_table: [num_blocks] int32 (this request's pages);
    kv_len: total context tokens (past + this chunk);
    q_start_pos: absolute position of q[0] (= kv_len - q_len).
    """
    q_len, hq, d = q.shape
    hkv = k_cache.shape[2]
    g = hq // hkv
    bs = k_cache.shape[1]
    dev = q.device
    seqused = torch.tensor([kv_len], dtype=torch.int32, device=dev)
    KTILE = block_n
    if int8_qk:
        n_tiles = triton.cdiv(kv_len, KTILE)
        if kmean is None:
            kmean = torch.empty(hkv, d, dtype=torch.float32, device=dev)
            _k_stats_kernel[(hkv,)](
                k_cache, block_table, kmean, seqused,
                k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
                block_table.stride(0) if block_table.dim() > 1 else 0,
                BLOCK_SIZE=bs, D=d, CHUNK=256, num_warps=4,
            )
        k8 = torch.empty(n_tiles * KTILE, hkv, d, dtype=torch.int8, device=dev)
        kscale = torch.empty(n_tiles, hkv, dtype=torch.float32, device=dev)
        _k_quant_kernel[(n_tiles, hkv)](
            k_cache, block_table, kmean, k8, kscale, seqused,
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
            k8.stride(0), k8.stride(1),
            HKV=hkv, BLOCK_SIZE=bs, D=d, TILE=KTILE, num_warps=4,
        )
    else:
        k8 = torch.empty(1, dtype=torch.int8, device=dev)
        kscale = torch.empty(1, dtype=torch.float32, device=dev)
    grid = (triton.cdiv(q_len, block_t), hq)
    _prefill_attn_kernel[grid](
        q, k_cache, v_cache, out, block_table, k8, kscale,
        seqused, q_start_pos, sm_scale,
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        k8.stride(0) if int8_qk else 0, k8.stride(1) if int8_qk else 0,
        out.stride(0), out.stride(1),
        0,
        q_len,
        HKV=hkv, KTILE=KTILE,
        G=g, D=d, BLOCK_SIZE=bs,
        BLOCK_T=block_t, BLOCK_N=block_n, INT8_QK=int8_qk,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out
