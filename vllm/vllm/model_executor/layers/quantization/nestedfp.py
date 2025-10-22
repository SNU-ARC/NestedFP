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
# import ipdb
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
        self.is_nestedfp_enabled = True
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
        current_nestedfp_enabled = self.is_nestedfp_enabled
        current_fp8_mode = self.fp8

        
        def fp16_to_fp8(x, dtype=torch.float8_e4m3fn):
            finfo = torch.finfo(torch.float8_e4m3fn)
            
            if x.dim() >= 2:
                # Per-token scaling: 각 토큰별로 독립적인 scale 계산
                x_abs_max = x.abs().amax(dim=-1, keepdim=True)
                
                # Outlier 처리: 99.9% percentile clipping
                if x_abs_max.numel() > 1:
                    # quantile 함수를 위해 float32로 변환
                    percentile_max = torch.quantile(x_abs_max.flatten().float(), 0.999)
                    x_abs_max = torch.minimum(x_abs_max, percentile_max.to(x_abs_max.dtype))
                
                # Per-token scale 계산
                scale = finfo.max / x_abs_max.clamp(min=1e-12)
            else:
                # 1D tensor의 경우 global scaling
                scale = finfo.max / x.abs().max().clamp(min=1e-12)
            
            # Scaling 및 saturation
            x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
            x8 = x_scl_sat.to(torch.float8_e4m3fn)
            
            # scale_factor를 원래 x의 dtype으로 유지
            scale_factor = scale.reciprocal().to(x.dtype)
            
            return x8, scale_factor
        
        def fp16_to_fp8_stable(x, dtype=torch.float8_e4m3fn):
            finfo = torch.finfo(torch.float8_e4m3fn)
            
            if x.dim() >= 2:
                x_abs_max = x.abs().amax(dim=-1, keepdim=True)
                
                # NaN/Inf 체크 (주석 처리된 상태 유지)
                #if torch.any(torch.isnan(x_abs_max)) or torch.any(torch.isinf(x_abs_max)):
                #    print("WARNING: NaN or Inf detected in x_abs_max!")
                #    x_abs_max = torch.nan_to_num(x_abs_max, nan=1.0, posinf=1.0, neginf=1.0)
                
                # 극단적으로 작은 값들을 더 안전하게 처리
                x_abs_max = torch.where(x_abs_max < 1e-5, 
                                    torch.full_like(x_abs_max, 1e-5), 
                                    x_abs_max)
                
                # Conservative scaling
                scale = (finfo.max * 0.7) / x_abs_max
                
                # Scale 범위 강제 제한
                scale = torch.clamp(scale, min=0.1, max=50.0)
                
            else:
                x_abs_max = x.abs().max()
                x_abs_max = max(x_abs_max.item(), 1e-5)
                scale = (finfo.max * 0.7) / x_abs_max
                scale = torch.clamp(scale, min=0.1, max=50.0)
            
            x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
            x8 = x_scl_sat.to(dtype)
            scale_factor = scale.reciprocal().to(x.dtype)
            
            return x8, scale_factor
                
        # current_fp8_mode와 current_nestedfp_enabled 사용
        if current_fp8_mode:
            if current_nestedfp_enabled:
                x8, scale_factor = fp16_to_fp8_stable(x)
                M = x8.shape[0]
                N = layer.weight.upper_part.shape[0]
                K = layer.weight.upper_part.shape[1]
                return (torch.ops.nestedfp.fp8_custom(M, N, K, layer.weight.upper_part, x8)) * scale_factor
            else:
                return F.linear(x, layer.weight, bias)
        else:
            if current_nestedfp_enabled:
                M = x.shape[0]
                N = layer.weight.upper_part.shape[0]
                K = layer.weight.upper_part.shape[1]
                output = torch.ops.nestedfp.fp16_custom(M, N, K, layer.weight.upper_part, layer.weight.lower_part, x)
                return output
            else:
                return F.linear(x, layer.weight, bias)
        
