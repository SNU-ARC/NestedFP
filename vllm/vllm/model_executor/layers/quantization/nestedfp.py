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

from vllm.model_executor.layers.quantization.utils.nestedfp_utils import * # nestedfp custom ops 등록



class NestedFPConfig(QuantizationConfig):
    """Config for Nested FP quantizer. It divides the fp 16 parameter to a upper part and lower part.
    
    Args: 
       None
    """

    def __init__(
        self,
        enable_nestedfp: bool = True,
    ) -> None:
        self.enable_nestedfp = enable_nestedfp
        return 

    @classmethod
    def get_name(cls) -> str:
        return "NestedFP"

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "NestedFPConfig":
        return cls()

    def get_linear_method(self) -> "NestedFPLinearMethod":
        return NestedFPLinearMethod(self)

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
                         prefix: str) -> Optional["NestedFPLinearMethod"]:
        
        with open("prefix_log.txt", "a") as f:
            f.write(f"Prefix: {prefix}\n")
            
        if "lm_head" in prefix:
            return None
        if isinstance(layer, LinearBase):
          return NestedFPLinearMethod(self)
        return None







class NestedFPLinearMethod(LinearMethodBase):
    """Linear method for Nested Fp quantizer

    Args:
        quant_config: the NestedFP quantization config.
    """

    def __init__(self, quant_config: NestedFPConfig):
        self.quant_config = quant_config
        self.weight = None
        self.is_nestedfp_enabled = False
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
        
        def fp16_to_fp8(x: torch.Tensor):
            dtype = torch.float8_e4m3fn
            finfo_max = 448.0
            finfo_min = -448.0
            min_denominator = 1e-12  # zero-division 방지용

            scale = finfo_max / x.abs().max().clamp(min=min_denominator)
            x_scl_sat = (x * scale).clamp(min=finfo_min, max=finfo_max)
            x8 = x_scl_sat.to(dtype)
            scale_factor = scale.float().reciprocal()

            return x8, scale_factor

        

        if self.fp8 is True:
            if self.is_nestedfp_enabled:
                # Per Tensor Quantization
                x8, scale_factor = fp16_to_fp8(x)
                dtype = torch.float8_e4m3fn
                finfo_max = 448.0
                finfo_min = -448.0
                min_denominator = 1e-12  # zero-division 방지용

                scale = finfo_max / x.abs().max().clamp(min=min_denominator)
                x_scl_sat = (x * scale).clamp(min=finfo_min, max=finfo_max)
                x8 = x_scl_sat.to(dtype)
                scale_factor = scale.float().reciprocal()
                
                
                M = x8.shape[0]
                N = layer.weight.upper_part.shape[0]
                K = layer.weight.upper_part.shape[1]
                return (torch.ops.nestedfp.fp8_custom(M, N, K, layer.weight.upper_part, x8)) * scale_factor
                
                
                # Per Token Quantization
                # dtype = torch.float8_e4m3fn
                # finfo_max = 448.0
                # finfo_min = -448.0
                # min_denominator = 1e-12  # zero-division 방지용

                # # x: [M, K] (M = batch_size × seq_len, K = hidden_dim)
                # absmax = x.abs().max(dim=1, keepdim=True)[0]              # [M, 1]
                # scale = finfo_max / absmax.clamp(min=min_denominator)     # [M, 1]
                # x_scl_sat = (x * scale).clamp(min=finfo_min, max=finfo_max)
                # x8 = x_scl_sat.to(dtype)                                  # [M, K]
                # scale_factor = scale.half().reciprocal()  # float16 reciprocal

                # M = x8.shape[0]
                # N = layer.weight.upper_part.shape[0]
                # K = layer.weight.upper_part.shape[1]

                # # 결과: [M, N], scale_factor: [M, 1] → broadcast 곱셈
                # output = torch.ops.nestedfp.fp8_custom(M, N, K, layer.weight.upper_part, x8)  # [M, N]
                # return output * scale_factor  # [M, N] × [M, 1]

            else:
                return F.linear(x, layer.weight, bias)
        else:
            if self.is_nestedfp_enabled:
                
                M = x.shape[0]
                N = layer.weight.upper_part.shape[0]
                K = layer.weight.upper_part.shape[1]
                output = torch.ops.nestedfp.fp16_custom(M, N, K, layer.weight.upper_part, layer.weight.lower_part, x)
                
                return output
            
            
            else:
                return F.linear(x, layer.weight, bias)
                
                
        
        
    
        
                    
            
            

            
            