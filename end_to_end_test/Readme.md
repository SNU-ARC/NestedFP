# VLLM Test Repository

This repository contains two primary components:
- `offline`
- `online`

---

## 1. offline

The `offline` folder contains scripts to measure **end-to-end throughput** of vLLM across multiple models and quantization modes.

### Supported Models:
- LLaMA 3.1 8B
- Mistral Nemo
- Mistral Small
- Phi-4

### Supported Precision Modes:
- FP16
- FP8
- NestedFP16
- NestedFP8

### Test Description:
- For each model and precision mode, throughput is measured by varying the batch size across different (input tokens, output tokens) combinations.
- The tests help assess how model performance scales with workload size under different quantization settings.

### Enabling NestedFP8 Mode:
To enable `nestedfp8` mode, you **must manually edit** the following file in the vLLM codebase:

```python
# vllm/vllm/model_executor/layers/quantization/dualfp.py
self.fp8 = True
Make sure this is done after installing vLLM in editable mode (pip install -e .), or the change will not take effect.
```

## 2. online

The `online` folder provides tools to simulate **real-time request traces** and evaluate vLLM’s serving performance using TPOT (Token-Per-Output-Time).

### Overview:
This test setup models an online serving scenario using a trace file that represents a realistic stream of requests. It allows you to evaluate vLLM's responsiveness under load.

### How to Run:

#### Step 1: Start the vLLM Server

```bash
python vllm_simple_server.py
```

- In `vllm_simple_server.py`, specify the desired model and quantization mode (`FP16`, `FP8`, etc.) by setting the `model_path` and related options.

#### Step 2: Launch the Asynchronous Client
```bash
python vllm_simple_client.py
```

- Sends requests to the running server using the timing defined in the trace file.
- Records response times and calculates the average TPOT (Token-Per-Output-Time).

### Customization via Alpha
- In `vllm_simple_client.py`, you can adjust the **request pacing** using the `alpha` parameter:
  - `alpha > 1.0`: Increases request rate (faster replay)
  - `alpha < 1.0`: Decreases request rate (slower replay)

---

## Notes

**Environment:**
- All tests should be run in the same Conda environment where vLLM is installed.
- Editable installation (`pip install -e .`) is recommended for modifying kernel behavior.

**Hardware Compatibility:**
- Compatible with NVIDIA GPUs that support FP8 and CUDA 11+.
- Actual speedup depends on GPU architecture, model size, and workload patterns.

**Performance Evaluation:**
- Compare throughput in `offline` tests and TPOT in `online` tests across precision modes to evaluate quantization impact.
