# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project



import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    MARLIN_SUPPORTED_GROUP_SIZES,
    apply_gptq_marlin_linear,
    marlin_make_empty,
    marlin_pad_dim,
    marlin_pad_qweight,
    marlin_pad_scales,
    marlin_padded_nk,
    marlin_permute_bias,
    marlin_permute_scales,
    marlin_zero_points,
    query_marlin_supported_quant_types,
    unpack_cols,
)
from vllm.model_executor.parameter import BasevLLMParameter, permute_param_layout_
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

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
# Opt-in: VLLM_MARLIN_REPACK_STAGED=1 turns it on (the GA100 mining cards, where the
# alternative was a CPU fallback); unset or any other value keeps the stock repack.
_REPACK_STAGING: dict = {}


def _use_staged_repack(device: torch.device) -> bool:
    return envs.VLLM_MARLIN_REPACK_STAGED == "1"


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

        if c.act_type.itemsize == 1:
            return False, "Marlin does not support 8-bit activations"

        quant_types = query_marlin_supported_quant_types(c.zero_points)
        if c.weight_type not in quant_types:
            return (
                False,
                (
                    f"Quant type ({c.weight_type}) not supported by"
                    f" Marlin, supported types are: {quant_types}"
                ),
            )

        if c.group_size not in MARLIN_SUPPORTED_GROUP_SIZES:
            return (
                False,
                (
                    f"Group size ({c.group_size}) not supported by "
                    "Marlin, supported group sizes are: "
                    f"{MARLIN_SUPPORTED_GROUP_SIZES}"
                ),
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

        size_k, size_n = c.partition_weight_shape
        padded_n, padded_k = marlin_padded_nk(size_n, size_k, c.group_size)

        # Default name since marlin requires empty parameter for zp,
        # TODO: remove this requirement from marlin (allow optional tensors)
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
                    size_k=padded_k,
                    size_n=padded_n,
                    num_bits=c.weight_type.size_bits,
                )
                return x
            x.data = ops.gptq_marlin_repack(
                marlin_pad_qweight(
                    x.data.contiguous(), size_n, size_k, padded_n, padded_k
                ),
                size_k=padded_k,
                size_n=padded_n,
                num_bits=c.weight_type.size_bits,
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
            )
            return x

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
                ),
            )
        else:
            # Marlin takes an empty zp; a loaded unused one keeps its data.
            self._retire_unused_zp(layer)
            self.w_zp_name = "w_zp"
            replace_parameter(layer, self.w_zp_name, marlin_make_empty(device))
        self._transform_param(layer, self.w_q_name, transform_w_q)
        self._transform_param(layer, self.w_s_name, transform_w_s)

        if hasattr(layer, "bias") and layer.bias is not None:
            replace_parameter(
                layer,
                "bias",
                marlin_permute_bias(marlin_pad_dim(layer.bias, size_n, padded_n)),
            )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.config
        w_q, w_s, w_zp = self._get_weight_params(layer)
        return apply_gptq_marlin_linear(
            input=x,
            weight=w_q,
            weight_scale=w_s,
            weight_zp=w_zp,  # type: ignore
            workspace=None,
            wtype=c.weight_type,
            input_size_per_partition=c.partition_weight_shape[0],
            output_size_per_partition=c.partition_weight_shape[1],
            bias=bias,
        )
