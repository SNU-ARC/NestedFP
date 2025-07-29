from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter

from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.utils import set_weight_attrs

import cutlass
import ipdb
import traceback

import torch._dynamo

from vllm.model_executor.layers.quantization.utils.dualfp_utils import * # dualfp custom ops 등록



class DUALFPConfig(QuantizationConfig):
    """Config for Dual FP quantizer. It divides the fp 16 parameter to a upper part and lower part.
    
    Args: 
       None
    """

    def __init__(
        self,
        enable_dualfp: bool = True,
    ) -> None:
        self.enable_dualfp = enable_dualfp
        return 

    @classmethod
    def get_name(cls) -> str:
        return "DUALFP"

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DUALFPConfig":
        return cls()

    def get_linear_method(self) -> "DUALFPLinearMethod":
        return DUALFPLinearMethod(self)

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16]
      
    @classmethod
    # Need to figure it out
    def get_min_capability(cls) -> int:
        return 60

    @staticmethod
    def get_config_filenames() -> List[str]:
        return []

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["DUALFPLinearMethod"]:
        
        with open("prefix_log.txt", "a") as f:
            f.write(f"Prefix: {prefix}\n")
            
        if "lm_head" in prefix:
            return None
        if isinstance(layer, LinearBase):
          return DUALFPLinearMethod(self)
        return None







class DUALFPLinearMethod(LinearMethodBase):
    """Linear method for Dual Fp quantizer

    Args:
        quant_config: the DualFP quantization config.
    """

    def __init__(self, quant_config: DUALFPConfig):
        self.quant_config = quant_config
        self.weight = None
        self.is_dualfp_enabled = True
        self.fp8 = True

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: List[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        
        weight = Parameter(torch.empty(sum(output_partition_sizes),
                                       input_size_per_partition,
                                       dtype=params_dtype),
                           requires_grad=False)
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)
        

   
    def apply(self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        assert(bias is None)
        
        # 글로벌 상태에서 모드 가져오기
        current_dualfp_enabled = DualFPGlobalState.get_dualfp_mode()
        current_fp8_mode = DualFPGlobalState.get_fp8_mode()
        
        def fp16_to_fp8(x, dtype=torch.float8_e4m3fn):
            finfo = torch.finfo(torch.float8_e4m3fn)
            scale = finfo.max / x.abs().max().clamp(min=1e-12)
            x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
            x8 = x_scl_sat.to(torch.float8_e4m3fn)
            scale_factor = scale.float().reciprocal()
            return x8, scale_factor
        
        # self.fp8 → current_fp8_mode, self.is_dualfp_enabled → current_dualfp_enabled 로 변경
        if current_fp8_mode:
            if current_dualfp_enabled:
                x8, scale_factor = fp16_to_fp8(x)
                M = x8.shape[0]
                N = layer.weight.upper_part.shape[0]
                K = layer.weight.upper_part.shape[1]
                return (torch.ops.dualfp.fp8_custom(M, N, K, layer.weight.upper_part, x8)) * scale_factor
            else:
                return F.linear(x, layer.weight, bias)
        else:
            if current_dualfp_enabled:
                M = x.shape[0]
                N = layer.weight.upper_part.shape[0]
                K = layer.weight.upper_part.shape[1]
                output = torch.ops.dualfp.fp16_custom(M, N, K, layer.weight.upper_part, layer.weight.lower_part, x)
                return output
            else:
                return F.linear(x, layer.weight, bias)