# NestedFP: High-Performance, Memory-Efficient Dual-Precision Floating Point Support for LLMs [[Paper]](https://arxiv.org/abs/2506.02024)

> Overview:

---

## TL;DR (Quickstart)

```bash
# 1) Create env
conda create -n nestedfp python=3.11 -y
conda activate nestedfp

# 2) Install vLLM (pinned)
mkdir -p ~/NestedFP && cd ~/NestedFP

git clone https://github.com/vllm-project/vllm.git tmp_vllm
mkdir -p vllm && cp -r tmp_vllm/.git vllm/ && rm -rf tmp_vllm
cd vllm

git add . && git commit --allow-empty -m "nestedfp"
git branch install && git checkout install
# Known-good commit
git reset --hard 70fedd0f7954079ebee36a7ca834cdf2f3e5d568

VLLM_USE_PRECOMPILED=1 pip install --editable .
git checkout main

# 3) Build and install NestedFP kernels
cd ..
cd nestedfp
./run.sh

# 4) Accuracy eval (GPU 0)
./scripts/acc_eval.sh 0 /home/ubuntu/models/Mistral-Small-24B-Base-2501

# 5) End-to-end eval
python scripts/e2e_bench.py --nestedfp --model /home/ubuntu/models/Mistral-Small-24B-Base-2501 &> result.txt

# 6) Kernel search (single GPU)
./scripts/kernel/run_fp16_single.sh 5120 32768 0 32 2048
```

---

## NestedFP Modes

### NestedFP8 Mode (`self.fp8 = True`)

* Uses `torch.ops.dualfp.fp8_custom` operation
* Automatic FP16→FP8 conversion with dynamic scaling
* Optimized for high throughput scenarios

### NestedFP16 Mode (`self.fp8 = False`)

* Uses standard FP16 residual path
* To switch to **NestedFP16**, open the installed vLLM repository’s `dualfp.py` file (found under `~/NestedFP/vllm/...`) and change the line where `self.fp8 = True` to `self.fp8 = False`.
* After modifying, rebuild or re-run your benchmark; the system will automatically run in NestedFP16 mode.

---

## Repository Layout

```
NestedFP/
├── vllm/                       # vLLM source (pinned to known-good commit)
├── nestedfp/                   # NestedFP kernel sources and build scripts
│   ├── run.sh                  # Build script for NestedFP kernels
│   └── ...
└── scripts/
    ├── acc_eval.sh             # Accuracy evaluation wrapper
    ├── e2e_bench.py            # End-to-end benchmark (TTFT/TPOT/SLO)
    └── kernel/
        ├── run_fp16_single.sh  # FP16 kernel search (single GPU)
        ├── run_fp16_multi.sh   # FP16 kernel search (multi GPU)
        ├── run_fp8_single.sh   # FP8  kernel search (single GPU)
        └── run_fp8_multi.sh    # FP8  kernel search (multi GPU)
```

---

## Requirements & CUDA Notes

* **Python**: 3.11
* **CUDA**: Ubuntu 24.04 → 12.6+
* **vLLM precompiled wheels**: built for CUDA 12.4. Options:

  1. **Recommended:** build vLLM from source with your CUDA 12.6+
  2. **Alternate:** use `VLLM_USE_PRECOMPILED=1` (mixed 12.4–12.6 works but not ideal)

---

## Verification

```python
from vllm import LLM, SamplingParams
llm = LLM(model="/home/ubuntu/models/Mistral-Small-24B-Base-2501", dtype="float16")
out = llm.generate(["Hello NestedFP!"], SamplingParams(max_tokens=16, temperature=0))
print(out[0].outputs[0].text)
```

---

## Accuracy Evaluation

```bash
./scripts/acc_eval.sh 0 /home/ubuntu/models/Mistral-Small-24B-Base-2501
```

Output: `scripts/results/acc_eval/`

---

## End-to-End Benchmark

```bash
python scripts/e2e_bench.py --nestedfp --model /home/ubuntu/models/Mistral-Small-24B-Base-2501 &> result.txt
```

* `--nestedfp` enables NestedFP8; remove it for baseline FP16.
* Add `--outfile` to control CSV name; it auto-appends `nestedfp` when enabled.

---

## Kernel Search

### FP16 (single GPU)

```bash
./scripts/kernel/run_fp16_single.sh 5120 32768 0 32 2048
```

### FP8

```bash
./scripts/kernel/run_fp8_single.sh 5120 32768 0 32 2048
```

Results saved under `scripts/results/kernel_search/`.

---

## Tips

* No need to manually export PYTHONPATH; scripts prepend `../vllm` automatically.
* All logs are relative to `scripts/results/`.
* Deterministic tests: use `temperature=0`.

---

## Troubleshooting

* CUDA mismatch → rebuild vLLM with your system CUDA.
* `pip` not found → ensure `conda activate nestedfp`.
* `ImportError: vllm not found` → ensure editable install success.

---

## Reproducibility

* Record git commits for both repos.
* Log GPU/driver/CUDA/torch/vLLM versions.
* Save all raw logs.

---

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
