#!/bin/bash

tasks=(
minerva_math
bbh_zeroshot
leaderboard_mmlu_pro
)

for task in "${tasks[@]}"; do
    PYTHONPATH="/disk/revision/vllm:$PYTHONPATH" VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=$1 VLLM_USE_V1=1 VLLM_FLASH_ATTN_VERSION=3 HF_ALLOW_CODE_EVAL=1 lm_eval --model vllm --model_args pretrained=/disk/models/$2,tensor_parallel_size=1,add_bos_token=True,dtype="float16",quantization="dualfp",enforce_eager=True --tasks ${task} --batch_size auto --trust_remote_code --confirm_run_unsafe_code &> result_$2_OURS_${task}_vllm_1.txt
done

for task in "${tasks[@]}"; do
    PYTHONPATH="/disk/revision/vllm:$PYTHONPATH" VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=$1 VLLM_USE_V1=1 VLLM_FLASH_ATTN_VERSION=3 HF_ALLOW_CODE_EVAL=1 lm_eval --model vllm --model_args pretrained=/disk/models/$2,tensor_parallel_size=1,add_bos_token=True,dtype="float16",quantization="dualfp",enforce_eager=True --tasks ${task} --batch_size auto --trust_remote_code --confirm_run_unsafe_code &> result_$2_OURS_${task}_vllm_2.txt
done

for task in "${tasks[@]}"; do
    PYTHONPATH="/disk/revision/vllm:$PYTHONPATH" VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=$1 VLLM_USE_V1=1 VLLM_FLASH_ATTN_VERSION=3 HF_ALLOW_CODE_EVAL=1 lm_eval --model vllm --model_args pretrained=/disk/models/$2,tensor_parallel_size=1,add_bos_token=True,dtype="float16",quantization="dualfp",enforce_eager=True --tasks ${task} --batch_size auto --trust_remote_code --confirm_run_unsafe_code &> result_$2_OURS_${task}_vllm_3.txt
done

for task in "${tasks[@]}"; do
    PYTHONPATH="/disk/revision/vllm:$PYTHONPATH" VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=$1 VLLM_USE_V1=1 VLLM_FLASH_ATTN_VERSION=3 HF_ALLOW_CODE_EVAL=1 lm_eval --model vllm --model_args pretrained=/disk/models/$2,tensor_parallel_size=1,add_bos_token=True,dtype="float16",quantization="dualfp",enforce_eager=True --tasks ${task} --batch_size auto --trust_remote_code --confirm_run_unsafe_code &> result_$2_OURS_${task}_vllm_4.txt
done

for task in "${tasks[@]}"; do
    PYTHONPATH="/disk/revision/vllm:$PYTHONPATH" VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=$1 VLLM_USE_V1=1 VLLM_FLASH_ATTN_VERSION=3 HF_ALLOW_CODE_EVAL=1 lm_eval --model vllm --model_args pretrained=/disk/models/$2,tensor_parallel_size=1,add_bos_token=True,dtype="float16",quantization="dualfp",enforce_eager=True --tasks ${task} --batch_size auto --trust_remote_code --confirm_run_unsafe_code &> result_$2_OURS_${task}_vllm_5.txt
done
