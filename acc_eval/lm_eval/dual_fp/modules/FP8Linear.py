import torch
import torch.nn as nn
from quant import quant_e4m3, quant_e5m2, quant_e4m3_scale
import cutlass

def convert_e4m3(x):
    x_ = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    quant_e4m3(x, x_)
    return x_

def convert_e5m2(x):
    x_ = torch.empty_like(x, dtype=torch.float8_e5m2)
    quant_e5m2(x, x_)
    return x_

def convert_e4m3_scale(x):
    x_ = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    quant_e4m3_scale(x, x_)
    return x_

class FP8Linear(nn.Module):
    def __init__(
            self,
            w,
            name,
            mode
        ):
        super().__init__()
        self.weight = w
        self.name = name
        self.mode = mode
        
    def forward(self, x):
        m = x.shape[0]
        n = self.weight.shape[0]

        torch.cuda.set_device(x.device)

        if len(x.shape) == 3 : 
            m = x.shape[1]
            x = x.squeeze(0)
            
        if self.mode == "fp16" :
            y = cutlass.cutlass_tma_warp_specialized_cooperative_streamk_2_1_1_128_16_64(self.weight, x).view(m, n)
        elif self.mode == "e4m3" :
            x = convert_e4m3(x)
            y = cutlass.cutlass_tma_warp_specialized_cooperative_fp8_2_1_1_128_128_128(self.weight, x).view(m, n)
        elif self.mode == "e5m2" :
            x = convert_e5m2(x)
            y = cutlass.cutlass_tma_warp_specialized_cooperative_fp8_e5m2_2_1_1_128_128_128(self.weight, x).view(m, n)
        elif self.mode == "ours" :
            if self.weight.dtype == torch.float8_e4m3fn :
                x = convert_e4m3(x)
                y = cutlass.cutlass_tma_warp_specialized_cooperative_fp8_scale_2_1_1_128_128_128(self.weight, x).view(m, n)
            else :
                y = cutlass.cutlass_tma_warp_specialized_cooperative_2_1_1_128_128_64(self.weight, x).view(m, n)
        else :
            assert False

        return y
