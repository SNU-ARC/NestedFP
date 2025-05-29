#!/bin/bash

tasks=(
#minerva_math
bbh_zeroshot
#leaderboard_mmlu_pro
)

for task in "${tasks[@]}"; do
    VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=$1 VLLM_USE_V1=1 VLLM_FLASH_ATTN_VERSION=3 HF_ALLOW_CODE_EVAL=1 lm_eval --model vllm --model_args pretrained=/disk/models/$2-FP8-Dynamic-Half,tensor_parallel_size=2,add_bos_token=True --tasks ${task} --batch_size auto --trust_remote_code --confirm_run_unsafe_code &> dummy/result_$2-FP8-Dynamic-Half_${task}_vllm_1.txt
done
