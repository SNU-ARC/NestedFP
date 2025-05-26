import torch
import torch.nn as nn

from quant import quant_e4m3, quant_e5m2, quant_e4m3_scale
from .FP8Linear import FP8Linear
import cutlass

def print_free_mem(dev):
    t = torch.cuda.get_device_properties(dev).total_memory
    r = torch.cuda.memory_reserved(dev)
    print(f"{(t - r) / 1024 / 1024 / 1024} GB")

def make_fp8(
    module,
    mode,
    name=""
):
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + "." + attr if name != "" else attr

        if isinstance(tmp, nn.Linear):
            if "norm" not in name1 and "embed" not in name1 and "head" not in name1 :
                assert tmp.bias == None

                w_ = tmp.weight
                torch.cuda.set_device(w_.device)

                if mode == "fp16" :
                    w = w_
                elif mode == "e4m3" :
                    w = torch.empty_like(w_, dtype=torch.float8_e4m3fn)
                    quant_e4m3(w_, w)
                elif mode == "e5m2" :
                    w = torch.empty_like(w_, dtype=torch.float8_e5m2)
                    quant_e5m2(w_, w)
                elif mode == "ours" :
                    count = torch.where(w_.abs() > 1.75, 1, 0).sum().item() != 0

                    if count == 0 :
                        w = torch.empty_like(w_, dtype=torch.float8_e4m3fn)
                        quant_e4m3_scale(w_, w)
                    #else :
                    #    w = torch.empty_like(w_, dtype=torch.float8_e5m2)
                    #    quant_e5m2(w_, w)
                else :
                    assert False

                delattr(module, attr)
                setattr(
                    module,
                    attr,
                    FP8Linear(
                        w,
                        name1,
                        mode
                    ),
                )

    for name1, child in module.named_children():
        make_fp8(
            child,
            mode,
            name + "." + name1 if name != "" else name1
        )
