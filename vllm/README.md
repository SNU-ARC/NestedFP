# DualFP: Dynamic Precision Quantization for vLLM

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/source/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM with DualFP" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/source/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
High-Performance LLM Serving with Adaptive Precision Quantization
</h3>

---

## Overview

DualFP is an advanced quantization configuration built on top of vLLM that introduces **NestedFP** - a dynamic precision quantization approach that can switch between FP16 and FP8 precision at runtime. This enables adaptive performance optimization based on server load and precision requirements.

## Key Features

- **Dynamic Precision Switching**: Runtime switching between FP16 (NestedFP16) and FP8 (NestedFP8) quantization
- **Optimized CUDA Kernels**: Custom CUTLASS-based kernels optimized for different matrix shapes
- **Model-Aware Optimization**: Specialized kernel selection for popular models (Llama 3.1, Mistral, Phi-4)
- **Scheduler Interface**: Ready for integration with adaptive scheduling systems
- **Zero-Overhead Switching**: Minimal performance impact when changing precision modes

## Architecture

### DualFP Linear Method

The core of DualFP is the `DualFPLinearMethod` class that replaces standard PyTorch matrix multiplications in Linear layers (QKV projections, MLP layers) with optimized implementations:

```python
class DUALFPLinearMethod(LinearMethodBase):
    def __init__(self, quant_config: DUALFPConfig):
        self.quant_config = quant_config
        self.weight = None
        self.is_dualfp_enabled = False
        self.fp8 = True  # Switch between FP8 (True) and FP16 (False)
```

### Precision Modes

**NestedFP8 Mode (`self.fp8 = True`)**:
- Uses `torch.ops.dualfp.fp8_custom` operation
- Automatic FP16→FP8 conversion with dynamic scaling
- Optimized for high throughput scenarios

**NestedFP16 Mode (`self.fp8 = False`)**:
- Uses `torch.ops.dualfp.fp16_custom` operation  
- Maintains FP16 precision throughout computation
- Optimized for scenarios requiring higher precision

### Kernel Optimization

DualFP includes 100+ optimized CUTLASS kernels automatically selected based on matrix dimensions:

```python
# Example kernel selection for Llama 3.1 8B
if (N, K) == (4096, 4096):
    range_kernel_map = [
        (32, "custom_5"),
        (64, "custom_6"), 
        (128, "custom_8"),
        (256, "custom_10"),
        (512, "custom_13"),
        (1024, "custom_48"),
        # ...
    ]
```

Specialized optimizations are provided for:
- **Llama 3.1 8B**: (4096,4096), (6144,4096), (28672,4096), (4096,14336)
- **Mistral Small**: (5120,4096), (5120,32768), (6144,5120), (65536,5120)  
- **Mistral Nemo**: (5120,14336), (28672,5120)
- **Phi-4**: (5120,5120), (5120,17920), (7680,5120), (35840,5120)

## Installation

### Prerequisites
- CUDA-compatible GPU
- Python 3.8+
- PyTorch with CUDA support
- CUTLASS library for optimized kernels

### Setup

1. **Clone and install vLLM in editable mode**:
```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 pip install --editable .
```

2. **Replace with DualFP code**:
Replace the vLLM installation with our DualFP-enhanced version (will be available via GitHub clone in the future).

3. **Install CUTLASS kernels**:
Install the cutlass library with NestedFP kernels

## Usage

### Basic Configuration

```python
from vllm import LLM, SamplingParams
from vllm.model_executor.layers.quantization import DUALFPConfig

# Configure DualFP quantization
quantization_config = DUALFPConfig()

# Initialize LLM with DualFP
llm = LLM(
    model="meta-llama/Llama-3.1-8B",
    quantization="dualfp"
)

# Generate text
prompts = ["Hello, how are you?"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
outputs = llm.generate(prompts, sampling_params)
```

### Runtime Precision Control

```python
# Switch to FP8 mode for higher throughput
llm.quantization_method.fp8 = True

# Switch to FP16 mode for higher precision  
llm.quantization_method.fp8 = False
```

### Advanced Usage with Scheduler Interface

DualFP provides a ready-to-use interface for adaptive schedulers to dynamically adjust precision based on server load and requirements. Note: The adaptive scheduler is not currently provided - only the interface is available for future integration.

```python
class AdaptiveScheduler:
    def adjust_precision(self, load_factor, latency_requirement):
        if load_factor > 0.8:
            # High load: switch to FP8 for throughput
            self.llm.quantization_method.fp8 = True
        elif latency_requirement < 100:  # ms
            # Low latency requirement: use FP16 for precision
            self.llm.quantization_method.fp8 = False
```


## Performance Benefits

### Throughput Improvements
- **NestedFP8 Mode**: Up to 1.55x speedup over standard FP16 implementations
- **NestedFP16 Mode**: Under 5% overehad over standard FP16 implementations on average

### Adaptive Performance
- **Load-Based Switching**: Automatically optimize for current server conditions
- **Zero-Overhead Transitions**: Precision changes take effect immediately without model reloading
- **Precision-Throughput Trade-off**: Fine-grained control over the accuracy vs speed balance

## Supported Models

DualFP has been optimized and tested with:

- **Llama Family**: Llama 3.1 8B, 70B, and variants
- **Mistral Family**: Mistral Small, Mistral Nemo-Base  
- **Phi Family**: Phi-4 and variants
- **General Support**: Any transformer-based model supported by vLLM

## File Structure

```
vllm/vllm/model_executor/layers/quantization/
├── dualfp.py                    # Main DualFP configuration and linear method
└── utils/
    └── dualfp_utils.py         # Custom operations and kernel selection
```

### Key Components

- **`DUALFPLinearMethod`**: Core linear layer implementation with dynamic precision
- **`torch.ops.dualfp.fp16_custom`**: Optimized FP16 GEMM operations
- **`torch.ops.dualfp.fp8_custom`**: Optimized FP8 GEMM operations with automatic scaling
- **Kernel Selection**: Automatic optimal kernel selection based on matrix shapes


## License

DualFP is released under the same license as vLLM. See the original vLLM repository for license details.

---

**Note**: DualFP is built on top of vLLM and inherits all of vLLM's capabilities while adding dynamic precision quantization. For general vLLM usage and features, please refer to the [official vLLM documentation](https://docs.vllm.ai).