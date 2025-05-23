#!/bin/bash

tasks=(
minerva_math
leaderboard_mmlu_pro
bbh_zeroshot

arc_easy
arc_challenge
hellaswag
piqa
winogrande
leaderboard_musr
gpqa_main_n_shot
ruler
)

for task in "${tasks[@]}"; do
    PYTHONPATH="/disk/vllm:$PYTHONPATH" VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=2 VLLM_USE_V1=1 VLLM_FLASH_ATTN_VERSION=3 HF_ALLOW_CODE_EVAL=1 lm_eval --model vllm --model_args pretrained=/disk/models/Llama-3.1-8B,tensor_parallel_size=1,add_bos_token=True,dtype="float16",quantization="dualfp",enforce_eager=True --tasks ${task} --batch_size auto --trust_remote_code --confirm_run_unsafe_code &> result_Llama-3.1-8B_OURS_${task}_vllm.txt
done
