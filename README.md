## To install dualfp library

Refer README.md at dualfp directory

## To run accuracy evaluation

PYTHONPATH="$(pwd)/vllm:$PYTHONPATH" VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=1 VLLM_FLASH_ATTN_VERSION=3 HF_ALLOW_CODE_EVAL=1 lm_eval --model vllm --model_args pretrained=/home/snu_arclab_2nd/models/Mistral-Small-24B-Base-2501,tensor_parallel_size=1,add_bos_token=True,dtype="float16",quantization="dualfp" --tasks bbh_zeroshot --batch_size auto --trust_remote_code --confirm_run_unsafe_code &> result.txt

## To run kernel evaluation

Refer README.md at kernel_eval directory

## To run end to end evaluation

Refer README.md at end_to_end_test directory

## Code bases
- cutlass directory (Kernel modifications)
- vllm directory (vLLM integration)

Please install conda env from dualfp.yml
