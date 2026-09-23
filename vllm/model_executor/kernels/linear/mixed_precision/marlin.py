# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project



import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    MARLIN_SUPPORTED_GROUP_SIZES,
    apply_gptq_marlin_linear,
    check_marlin_supports_shape,
    marlin_act_int8_process_scales,
    marlin_is_k_full,
    marlin_make_empty_g_idx,
    marlin_make_workspace_new,
    marlin_pad_dim,
    marlin_pad_qweight,
    marlin_pad_scales,
    marlin_padded_nk,
    marlin_permute_bias,
    marlin_permute_scales,
    marlin_sort_g_idx,
    marlin_zero_points,
    query_marlin_supported_quant_types,
    unpack_cols,
)
from vllm.model_executor.parameter import BasevLLMParameter, permute_param_layout_
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

# (syv) #27: sm80 (GA100) wedges with Xid 31 during Marlin repack -- not from
# the repack kernel, but from VMM mapping churn under the per-layer transient
# allocations around it (the contiguous copy and the pad), after which an
# unrelated elementwise kernel write-faults asynchronously. ahnguyen17 verified
# on a CMP 170HX that the same math staged through CPU is bit-exact and
# Xid-free, which convicts the allocation pattern, not the arithmetic. This is
# the GPU-resident version of that fix: pad/copy into ONE grow-only staging
# buffer per device (layer shapes repeat, so it reallocates a handful of times
# instead of alloc/freeing two transients per layer), hand the SAME values to
# the repack kernel, get bit-identical output. Costs one max-layer-sized buffer
# (~141 MB on this stack) that stays resident after load on the cards where it
# is on -- the card class where the alternative is a wedge-until-reboot.
# Default: on for compute capability 8.0 exactly, off elsewhere; override with
# VLLM_MARLIN_REPACK_STAGED=0/1.
_REPACK_STAGING: dict = {}


def _use_staged_repack(device: torch.device) -> bool:
    env = envs.VLLM_MARLIN_REPACK_STAGED
    if env is not None:
        return env == "1"
    if device.type != "cuda":
        return False
    return torch.cuda.get_device_capability(device) == (8, 0)


def _staged_pad_qweight(
    qweight: torch.Tensor, size_n: int, size_k: int, padded_n: int, padded_k: int
) -> torch.Tensor:
    """marlin_pad_qweight + .contiguous() through a persistent staging buffer.

    Returns a view holding exactly what the unstaged path would pass to
    gptq_marlin_repack; the caller must consume it before the next call.
    """
    pack_factor = size_k // qweight.size(0)
    rows, cols = padded_k // pack_factor, padded_n
    need = rows * cols
    buf = _REPACK_STAGING.get(qweight.device)
    if buf is None or buf.numel() < need or buf.dtype != qweight.dtype:
        _REPACK_STAGING.pop(qweight.device, None)
        buf = torch.empty(need, dtype=qweight.dtype, device=qweight.device)
        _REPACK_STAGING[qweight.device] = buf
    view = buf[:need].view(rows, cols)
    if (rows, cols) != (qweight.size(0), qweight.size(1)):
        view.zero_()
    view[: qweight.size(0), : qweight.size(1)].copy_(qweight)
    return view


from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig


class MarlinLinearKernel(MPLinearKernel):
    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        # Marlin uses inline PTX, so it can only be compatible with Nvidia
        if not current_platform.is_cuda():
            return False, "Marlin only supported on CUDA"

        quant_types = query_marlin_supported_quant_types(c.zero_points)
        if c.weight_type not in quant_types:
            return (
                False,
                f"Quant type ({c.weight_type}) not supported by"
                f"  Marlin, supported types are: {quant_types}",
            )

        if c.group_size not in MARLIN_SUPPORTED_GROUP_SIZES:
            return (
                False,
                f"Group size ({c.group_size}) not supported by "
                "Marlin, supported group sizes are: "
                f"{MARLIN_SUPPORTED_GROUP_SIZES}",
            )

        if c.has_g_idx:
            # Act-order couples K to the full-model group layout, so tile
            # padding is not supported; keep the strict shape check.
            return check_marlin_supports_shape(
                c.partition_weight_shape[1],  # out_features
                c.partition_weight_shape[0],  # in_features
                c.full_weight_shape[0],  # in_features
                c.group_size,
            )

        # A group straddling TP ranks cannot be fixed by padding.
        if (
            c.group_size != -1
            and c.group_size < c.full_weight_shape[0]
            and c.partition_weight_shape[0] % c.group_size != 0
        ):
            return False, (
                f"in_features per partition {c.partition_weight_shape[0]} is "
                f"not divisible by group_size = {c.group_size}."
            )

        # Tile misalignment is fixed by zero-padding at weight prep.
        return True, None

    # note assumes that
    #  `weight_packed` is: {input_dim = 0, output_dim = 1, packed_dim = 0}
    #  `weight_scale` is: {input_dim = 0, output_dim = 1}
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        device = getattr(layer, self.w_q_name).device
        c = self.config
        is_a_8bit = c.act_type is not None and c.act_type.itemsize == 1

        if is_a_8bit:
            # syv patch (marlin-int8-asym-zp): the kernel set instantiates
            # kS8 x kU4 (AWQ zero-point weights, int8 activations) next to
            # kS8 x kU4B8; only 8-bit weights have no int8-activation kernel.
            assert c.weight_type in (scalar_types.uint4b8, scalar_types.uint4), (
                "W8A8 is not supported by marlin kernel."
            )

        # syv patch: the int8-activation Marlin kernel reads the (int16-requantized)
        # group scales as *unsigned* int16 (marlin_template.h: reinterpret_cast<
        # uint16_t*>), so checkpoints with negative group scales (AutoRound
        # symmetric exports have ~50%) produce garbage. Fold the sign into the
        # int4 codes: s -> -s, q -> -q (uint4b8 code v -> 16 - v, clamped to 15;
        # v == 0 i.e. q == -8 loses one LSB, extremely rare).
        if (
            c.act_type == torch.int8
            and c.group_size != -1
            and not c.has_g_idx
            and not c.zero_points
        ):
            _wq = getattr(layer, self.w_q_name)
            _ws = getattr(layer, self.w_s_name)
            permute_param_layout_(_wq, input_dim=0, output_dim=1, packed_dim=0)
            permute_param_layout_(_ws, input_dim=0, output_dim=1)
            _neg = _ws.data < 0
            if bool(_neg.any()):
                _q = _wq.data
                _rows_per_group = c.group_size // 8
                _neg_rows = _neg.repeat_interleave(_rows_per_group, dim=0)[: _q.shape[0]]
                _shifts = torch.arange(0, 32, 4, device=_q.device, dtype=torch.int32)
                _nib = (_q.unsqueeze(-1) >> _shifts) & 0xF
                _flip = torch.clamp(16 - _nib, max=15)
                _nib = torch.where(_neg_rows.unsqueeze(-1), _flip, _nib)
                _out = torch.zeros_like(_q)
                for _i in range(8):
                    _out |= _nib[..., _i] << (4 * _i)
                _wq.data = _out
                _ws.data = torch.where(_neg, -_ws.data, _ws.data)
                del _nib, _flip, _out

        if c.act_type == torch.float8_e4m3fn:
            ops.marlin_int4_fp8_preprocess(getattr(layer, self.w_q_name), inplace=True)
            getattr(layer, self.w_s_name).data = (
                getattr(layer, self.w_s_name).data * 512
            )

        row_parallel = c.partition_weight_shape[0] != c.full_weight_shape[0]
        self.is_k_full = marlin_is_k_full(c.has_g_idx, row_parallel)

        size_k, size_n = c.partition_weight_shape
        if c.has_g_idx:
            # Act-order shapes were strictly validated in can_implement.
            padded_n, padded_k = size_n, size_k
        else:
            padded_n, padded_k = marlin_padded_nk(size_n, size_k, c.group_size)

        # Allocate marlin workspace, reusing existing storage on reload.
        self.workspace = marlin_make_workspace_new(
            device, existing=getattr(self, "workspace", None)
        )

        # Default names since marlin requires empty parameters for these,
        # TODO: remove this requirement from marlin (allow optional tensors)
        if self.w_gidx_name is None:
            self.w_gidx_name = "g_idx"
        if self.w_zp_name is None:
            self.w_zp_name = "w_zp"

        def transform_w_q(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)
            if _use_staged_repack(x.data.device):
                # (syv) #27: stage contiguous+pad through the persistent buffer
                # and drop the source before the repack output allocates, so
                # the only per-layer allocation left is the final weight.
                staged = _staged_pad_qweight(
                    x.data, size_n, size_k, padded_n, padded_k
                )
                x.data = torch.empty(0, dtype=staged.dtype, device=staged.device)
                x.data = ops.gptq_marlin_repack(
                    staged,
                    perm=layer.g_idx_sort_indices,
                    size_k=padded_k,
                    size_n=padded_n,
                    num_bits=c.weight_type.size_bits,
                    is_a_8bit=is_a_8bit,
                )
                return x
            x.data = ops.gptq_marlin_repack(
                marlin_pad_qweight(
                    x.data.contiguous(), size_n, size_k, padded_n, padded_k
                ),
                perm=layer.g_idx_sort_indices,
                size_k=padded_k,
                size_n=padded_n,
                num_bits=c.weight_type.size_bits,
                is_a_8bit=is_a_8bit,
            )
            return x

        def transform_w_s(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1)
            x.data = marlin_permute_scales(
                marlin_pad_scales(
                    x.data.contiguous(),
                    size_n,
                    size_k,
                    padded_n,
                    padded_k,
                    c.group_size,
                ),
                size_k=padded_k,
                size_n=padded_n,
                group_size=c.group_size,
                is_a_8bit=is_a_8bit,
            )

            if c.group_size == -1:
                num_groups = 1
            else:
                num_groups = c.partition_weight_shape[0] // c.group_size

            if c.act_type == torch.int8 and num_groups > 1:
                x.data, input_global_scale = marlin_act_int8_process_scales(x.data)
                layer.register_parameter(
                    "input_global_scale",
                    torch.nn.Parameter(input_global_scale, requires_grad=False),
                )
            else:
                layer.input_global_scale = None
            return x

        if c.has_g_idx:
            g_idx, g_idx_sort_indices = marlin_sort_g_idx(
                getattr(layer, self.w_gidx_name)
            )
            self._transform_param(layer, self.w_gidx_name, lambda _: g_idx)
            replace_parameter(
                layer, "g_idx_sort_indices", g_idx_sort_indices, prefer_copy=True
            )
        else:
            setattr(layer, self.w_gidx_name, marlin_make_empty_g_idx(device))
            layer.g_idx_sort_indices = marlin_make_empty_g_idx(device)

        if c.zero_points:
            grouped_k = size_k // c.group_size if c.group_size != -1 else 1
            padded_grouped_k = padded_k // c.group_size if c.group_size != -1 else 1
            self._transform_param(
                layer,
                self.w_zp_name,
                lambda x: marlin_zero_points(
                    marlin_pad_scales(
                        unpack_cols(
                            x.t(),
                            c.weight_type.size_bits,
                            grouped_k,
                            size_n,
                        ),
                        size_n,
                        size_k,
                        padded_n,
                        padded_k,
                        c.group_size,
                    ),
                    size_k=padded_grouped_k,
                    size_n=padded_n,
                    num_bits=c.weight_type.size_bits,
                    is_a_8bit=is_a_8bit,
                ),
            )
        else:
            setattr(layer, self.w_zp_name, marlin_make_empty_g_idx(device))
        self._transform_param(layer, self.w_q_name, transform_w_q)
        self._transform_param(layer, self.w_s_name, transform_w_s)

        if hasattr(layer, "bias") and layer.bias is not None:
            layer.bias.data = marlin_permute_bias(
                marlin_pad_dim(layer.bias, size_n, padded_n)
            )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.config
        w_q, w_s, w_zp, w_gidx = self._get_weight_params(layer)

        # `process_weights_after_loading` will ensure w_zp and w_gidx are not
        #  None for marlin

        return apply_gptq_marlin_linear(
            input=x,
            weight=w_q,
            weight_scale=w_s,
            weight_zp=w_zp,  # type: ignore
            g_idx=w_gidx,  # type: ignore
            g_idx_sort_indices=layer.g_idx_sort_indices,
            workspace=self.workspace,
            wtype=c.weight_type,
            input_size_per_partition=c.partition_weight_shape[0],
            output_size_per_partition=c.partition_weight_shape[1],
            is_k_full=self.is_k_full,
            input_global_scale=getattr(layer, "input_global_scale", None),
            bias=bias,
            input_dtype=c.act_type,
        )
