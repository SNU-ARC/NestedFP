# NestedFP: High-Performance, Memory-Efficient Dual-Precision Floating Point Support for LLMs [[Paper]](https://arxiv.org/abs/2506.02024)

## Overview

**NestedFP** is a high-performance, memory-efficient dual-precision framework for LLM serving that enables both FP8 and FP16 inference from a single 16-bit model without additional memory overhead. It introduces a lightweight FP16→(FP8 + residual) decomposition and a custom CUTLASS-based kernel integrated into vLLM, achieving up to 1.55× throughput improvement in FP8 mode with accuracy comparable to standard quantized FP8 models. It also retains full FP16 precision capability for dynamic, SLO-aware serving.

## Requirements

* **Ubuntu 22.04**
* **CUDA 12.4**

## Setup

```bash
# 1. Clone NestedFP repository
git clone https://github.com/SNU-ARC/NestedFP.git
cd NestedFP

# 2. Create environment
conda create -n nestedfp python=3.11 -y
conda activate nestedfp

# 3. Install vLLM 0.8.3 precompiled version
#    Clone vLLM into a temporary folder, then copy only the .git directory
mkdir -p tmp && cd tmp
git clone https://github.com/vllm-project/vllm.git
cd ..

cp -r tmp/vllm/.git vllm/
rm -rf tmp

cd vllm
git add .
git commit -m "nestedfp"
git branch install
git checkout install
git reset --hard 70fedd0f7954079ebee36a7ca834cdf2f3e5d568

# Install with precompiled CUDA 12.4 wheel (for Ubuntu 22.04)
VLLM_USE_PRECOMPILED=1 pip install --editable .

# Return to main branch
git checkout main

# 4. Install NestedFP kernels
cd ../nestedfp
./run.sh

# 5. Install lm-eval library
pip install lm-eval==0.4.8
```

## Repository Layout

```
NestedFP/
├── vllm/ # vLLM source with NestedFP modifications
├── cutlass/ # CUTLASS source with custom kernels
├── nestedfp/ # Python–C++ interface and build scripts for custom CUTLASS kernels
└── scripts/
├── acc_eval.sh # accuracy evaluation script
├── e2e_bench.py # end-to-end latency evaluation
└── kernel/
├── run_fp16_single.sh # FP16 kernel search (single GPU)
├── run_fp16_multi.sh # FP16 kernel search (multi GPU)
├── run_fp8_single.sh # FP8 kernel search (single GPU)
└── run_fp8_multi.sh # FP8 kernel search (multi GPU)
```

## NestedFP Modes

NestedFP supports two precision modes: **NestedFP16** and **NestedFP8**. You can switch between them by editing the following line in `NestedFP/vllm/vllm/model_executor/layers/quantization/dualfp.py` (line 91):

```python
self.fp8 = True  # Set to False for NestedFP16 mode
```

## Accuracy Evaluation

You can evaluate model accuracy using the following command:

```bash
./scripts/acc_eval.sh <GPU_ID> <MODEL_PATH>
```

- `<gpu_id>`: The GPU index to use
- `<model_path>`: Path to the model directory  

**Example:**
```bash
./scripts/acc_eval.sh 0 Mistral-Small-24B-Base-2501
```

**Output:**
All results will be saved to: `./results/acc_eval/`

## Kernel Search

You can search for the optimal CUTLASS kernel using the following command. The script sweeps over 80 candidate kernels to find the best-performing one for a specific GEMM shape. You can check the GEMM shapes used in our customized vLLM version at: `NestedFP/vllm/vllm/model_executor/layers/quantization/utils/dualfp_utils.py`

**Command format:**
```bash
./scripts/kernel/run_fp16_single.sh N K GPU M_START M_END  
./scripts/kernel/run_fp8_single.sh N K GPU M_START M_END  
```

- `GPU` — GPU index to use
- `M_START` — starting M dimension for the search range
- `M_END` — ending M dimension for the search range

**Example (FP16):**
```bash
./scripts/kernel/run_fp16_single.sh 5120 32768 0 32 2048
```

**Example (FP8):**
```bash
./scripts/kernel/run_fp8_single.sh 5120 32768 0 32 2048
```

## End-to-End Benchmark

You can run the end-to-end benchmark using the following command:
```bash
python scripts/e2e_bench.py --nestedfp --model <MODEL_PATH>
```

- `<MODEL_PATH>` — path to the model directory
- `--nestedfp` — enables NestedFP mode; remove this flag to run in baseline FP16 mode

**Example:**
```bash
python scripts/e2e_bench.py --nestedfp --model Mistral-Small-24B-Base-2501
```

**Output:**

All benchmark logs and results will be saved to: `./scripts/results/e2e/`

## Citation

Please cite our paper if you find our work useful:

```bibtex
@inproceedings{lee2025nestedfp,
  title={NestedFP: High-Performance, Memory-Efficient Dual-Precision Floating Point Support for LLMs},
  author={Haeun Lee and Omin Kwon and Yeonhong Park and Jae W. Lee},
  year={2025},
  booktitle={Proceedings of the 39th Conference on Neural Information Processing Systems}
}
```
