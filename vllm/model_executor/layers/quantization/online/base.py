# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch

from vllm.config.quantization import QuantizationConfigArgs, QuantSpec
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    RoutedExperts,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    should_ignore_layer,
)
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PerBlockOnlineLinearMethod,
    Fp8PerBlockOnlineMoEMethod,
    Fp8PerTensorOnlineLinearMethod,
    Fp8PerTensorOnlineMoEMethod,
)
from vllm.model_executor.layers.quantization.online.int8 import (
    Int8OnlineMoEMethod,
)
from vllm.model_executor.layers.quantization.online.mxfp8 import (
    Mxfp8OnlineLinearMethod,
    Mxfp8OnlineMoEMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Static128BlockSym,
    kFp8StaticTensorSym,
    kInt8StaticChannelSym,
    kMxfp8Dynamic,
)

logger = init_logger(__name__)


# Online dispatch tables, keyed by the QuantSpec.weight QuantKey. The
# corresponding method class handles the activation choice via its
# `supported_activation_quant` set.
_ONLINE_LINEAR_METHODS: dict[QuantKey, type] = {
    kFp8StaticTensorSym: Fp8PerTensorOnlineLinearMethod,
    kFp8Static128BlockSym: Fp8PerBlockOnlineLinearMethod,
    kMxfp8Dynamic: Mxfp8OnlineLinearMethod,
}

_ONLINE_MOE_METHODS: dict[QuantKey, type] = {
    kFp8StaticTensorSym: Fp8PerTensorOnlineMoEMethod,
    kFp8Static128BlockSym: Fp8PerBlockOnlineMoEMethod,
    kMxfp8Dynamic: Mxfp8OnlineMoEMethod,
    kInt8StaticChannelSym: Int8OnlineMoEMethod,
}


def _sm70_qpn8_available() -> bool:
    """True when the Volta build carries the packed QPN8 kernels."""
    from vllm.platforms import current_platform

    if not current_platform.is_cuda():
        return False
    capability = current_platform.get_device_capability()
    if capability is None or capability.major != 7:
        return False
    try:
        import vllm._C  # noqa: F401
    except ImportError:
        return False
    return hasattr(torch.ops._C, "fp8_qpn8_prepare_sm70") and hasattr(
        torch.ops._C, "fp8_qpn8_dispatch_sm70_out"
    )


class OnlineQuantizationConfig(QuantizationConfig):
    """Model-level config for online quantization (quantize fp16/bf16 weights
    during model loading, without requiring a pre-quantized checkpoint)."""

    def __init__(
        self,
        args: QuantizationConfigArgs,
    ) -> None:
        super().__init__()
        if args.linear is None and args.moe is None:
            raise ValueError(
                "OnlineQuantizationConfig requires at least one of "
                "quantization_config.linear or quantization_config.moe "
                "to be set."
            )
        self.args = args
        self.ignored_layers: list[str] = args.ignore

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "online"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        # Note: as more online quant schemes will be added, this
        # value will become the minimum across all supported schemes.
        #
        # Volta is admitted for the blockwise FP8 scheme, because the SM70 build
        # carries the packed QPN8 kernels (fp8_qpn8_prepare_sm70 and
        # fp8_qpn8_dispatch_sm70_out) that replace the SM75+ paths. Without those
        # operators the SM70 route does not exist and the ordinary floor applies.
        if _sm70_qpn8_available():
            return 70
        return 75


    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "OnlineQuantizationConfig":
        raise NotImplementedError(
            "OnlineQuantizationConfig does not support loading from a "
            "checkpoint config. Use quantization_config or "
            "quantization='fp8_per_tensor'/'fp8_per_block' instead."
        )

    def _dispatch(
        self,
        spec: QuantSpec | None,
        table: dict[QuantKey, type],
        layer: torch.nn.Module,
    ) -> "QuantizeMethodBase | None":
        if spec is None or spec.weight is None:
            return None
        cls = table.get(spec.weight)
        if cls is None:
            raise ValueError(
                f"online quantization for {type(layer).__name__} with "
                f"weight={spec.weight} is not supported; supported weight "
                f"keys: {sorted(str(k) for k in table)}"
            )
        # Online method classes pick their own activation format internally.
        # Per-class activation overrides are not yet wired through; reject
        # explicit overrides until the relevant method class opts in.
        if spec.activation is not None:
            raise ValueError(
                f"activation override (activation={spec.activation}) is not "
                f"yet supported for online {cls.__name__}"
            )
        if isinstance(layer, RoutedExperts):
            return cls(layer=layer)
        return cls()

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        if isinstance(layer, LinearBase):
            if should_ignore_layer(
                prefix,
                ignore=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            method = self._dispatch(self.args.linear, _ONLINE_LINEAR_METHODS, layer)
            return method if method is not None else UnquantizedLinearMethod()
        elif isinstance(layer, RoutedExperts):
            if should_ignore_layer(
                prefix,
                ignore=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            method = self._dispatch(self.args.moe, _ONLINE_MOE_METHODS, layer)
            return (
                method
                if method is not None
                else UnquantizedFusedMoEMethod(layer.moe_config)
            )
        return None
