## To run accuracy evaluation

For all evaluations, model name and GPU number need to be set. 
We measured baseline accuracy with recent vLLM version. 
Please install conda env from vllm_fp8.yml.
For our scheme, please install conda env from dualfp.yml.

```bash
./evaluate_vllm_fp16.sh GPU_NUMBER MODEL_NAME # To run fp16 model
./evaluate_vllm_fp8.sh GPU_NUMBER MODEL_NAME # To run vLLM quantized FP8 model
./evaluate_vllm_nested_fp8.sh GPU_NUMBER MODEL_NAME # To run NestFP8 model
```

In order to run vLLM fp8 model, offline quantization prior to execution is necessary. 

```bash
python run_quant_vllm.py
```
