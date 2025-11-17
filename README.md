# NestedFP: High-Performance, Memory-Efficient Dual-Precision Floating Point Support for LLMs [[Paper]](https://arxiv.org/abs/2506.02024)

## Overview

**NestedFP** is a high-performance, memory-efficient dual-precision framework for LLM serving that supports both FP8 and FP16 inference from a single FP16 model without additional memory overhead. It introduces a lightweight FP16 → (FP8 + residual) decomposition and CUTLASS-based custom kernels integrated into vLLM, delivering FP8 accuracy on par with standard quantized FP8 models while preserving full FP16 precision. NestedFP further enables dynamic, SLO-aware serving by allowing runtime precision selection.

## Requirements

- **Ubuntu 22.04**
- **GCC/G++ 12.3.0**
- **CUDA 12.6**

### Optional
```bash
sudo apt install ninja-build  # For faster compilation
```

## Setup

Run the following command to install NestedFP and its dependencies:

```bash
./install.sh
```

## Repository Layout
```
NestedFP/
├── vllm/                      # vLLM source with NestedFP modifications
├── cutlass/                   # CUTLASS source with custom kernels
├── nestedfp/                  # Python–C++ interface and build scripts for custom CUTLASS kernels
├── scripts/
│   ├── acc_eval.sh            # accuracy evaluation script
│   ├── vllm_simple_server.py  # vLLM server launcher for streaming requests
│   ├── vllm_simple_client.py  # vLLM client for sending requests
│   └── kernel/
│       ├── run_fp16_single.sh # FP16 kernel search (single GPU)
│       ├── run_fp16_multi.sh  # FP16 kernel search (multi GPU)
│       ├── run_fp8_single.sh  # FP8 kernel search (single GPU)
│       └── run_fp8_multi.sh   # FP8 kernel search (multi GPU)
└── example/                   # example usage scripts
```

## Precision Mode Configuration

NestedFP requires explicitly selecting the precision mode before each experiment by modifying two files:
- `NestedFP/vllm/vllm/model_executor/layers/quantization/nestedfp.py`
- `NestedFP/vllm/vllm/v1/core/sched/scheduler.py`

| Mode | nestedfp.py<br>Line 93 | nestedfp.py<br>Line 95 | scheduler.py<br>Lines 445-447 |
|------|:---:|:---:|:---:|
| **FP8** | ✅ Uncomment | ❌ Comment | ❌ Comment |
| **FP16** | ❌ Comment | ✅ Uncomment | ❌ Comment |
| **Dynamic Precision Selection** | ❌ Comment | ❌ Comment | ✅ Uncomment |

> **Note:** Only one precision mode can be active at a time.

## Accuracy Evaluation

### Running Accuracy Evaluation

For accuracy evaluation, configure the precision mode to **FP8** (see [Precision Mode Configuration](#precision-mode-configuration)).

**Command Format:**
```bash
./scripts/acc_eval.sh <GPU_ID> <MODEL_PATH> --nestedfp
```

**Parameters:**
- `<GPU_ID>` — GPU index to use for evaluation
- `<MODEL_PATH>` — Path to the model directory

### Example
```bash
./scripts/acc_eval.sh 0 Mistral-Small-24B-Base-2501 --nestedfp
```

### Output

All evaluation results will be saved to: `./results/acc_eval/`

## Kernel Search

### Overview

The kernel search script benchmarks 80 candidate CUTLASS kernels for each target GEMM configuration (including batch size) on NVIDIA H100 GPUs and reports their performance for manual kernel selection.

**Optimal kernels for each GEMM shape:**  
You can find the reference mapping from GEMM shapes to their optimal CUTLASS kernels in our customized vLLM at: `NestedFP/vllm/vllm/model_executor/layers/quantization/utils/nestedfp_utils.py`

### Running Kernel Search

For kernel search, configure the precision mode to **FP16** (see [Precision Mode Configuration](#precision-mode-configuration)).

**Command Format:**
```bash
./scripts/kernel/run_fp16_single.sh N K GPU M_START M_END
```

**Parameters:**
- `N` — N dimension of the GEMM shape
- `K` — K dimension of the GEMM shape
- `GPU` — GPU index to use for the search
- `M_START` — Starting M dimension for the search range
- `M_END` — Ending M dimension for the search range

### Example

**FP16 Kernel Search:**
```bash
./scripts/kernel/run_fp16_single.sh 5120 32768 0 32 2048
```

This command searches for the optimal kernel with:
- N = 5120, K = 32768
- GPU 0
- M dimension range: 32 to 2048 (in steps of 32)

## Throughput Test

### Running the Test

For throughput test, configure the precision mode to **FP16** (see [Precision Mode Configuration](#precision-mode-configuration)).

#### 1. Start the Model Server

First, load your model using the following command:
```bash
python scripts/vllm_simple_server.py \
  --model <MODEL_PATH> \
  --max-num-batched-tokens 8192 \
  --port 8000
```

**Parameters:**
- `<MODEL_PATH>` — Path to the model directory
- `--quantization nestedfp` — Enables NestedFP mode (omit this flag for baseline FP16 mode)

When the model loads successfully, you'll see:
```
INFO:     Started server process [200291]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

#### 2. Run the Throughput Test

In a separate terminal, execute:
```bash
python scripts/vllm_simple_client.py \
  --model <MODEL_PATH> \
  --api-url http://0.0.0.0:8000/v1/completions \
  --test-mode throughput
```

### Examples

**Vanilla FP16 Execution:**
```bash
# Start server
python scripts/vllm_simple_server.py \
  --model Mistral-Small-24B-Base-2501 \
  --max-num-batched-tokens 8192 \
  --port 8000

# Run client
python scripts/vllm_simple_client.py \
  --model Mistral-Small-24B-Base-2501 \
  --api-url http://0.0.0.0:8000/v1/completions \
  --test-mode throughput
```

**NestedFP FP16 Execution:**
```bash
# Start server with NestedFP
python scripts/vllm_simple_server.py \
  --model Mistral-Small-24B-Base-2501 \
  --max-num-batched-tokens 8192 \
  --port 8000 \
  --quantization nestedfp

# Run client with NestedFP
python scripts/vllm_simple_client.py \
  --model Mistral-Small-24B-Base-2501 \
  --api-url http://0.0.0.0:8000/v1/completions \
  --test-mode throughput \
  --nestedfp
```

### Additional Options

You can customize the test parameters by passing additional options to `vllm_simple_client.py`:
- Input/output token length
- Batch size

See the script's help documentation for all available options.

### Notes

For stable performance measurements, we recommend running each test at least twice.

## Citation

Please cite our paper if you find our work useful:

```bibtex
@inproceedings{lee2025nestedfp,
  title={NestedFP: High-Performance, Memory-Efficient Dual-Precision Floating Point Support for LLMs},
  author={Haeun Lee and Omin Kwon and Yeonhong Park and Jae W. Lee},
  year={2025},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems}
}
```
