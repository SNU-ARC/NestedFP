#!/bin/bash

tasks=(
minerva_math
bbh_zeroshot
#leaderboard_mmlu_pro
)

for task in "${tasks[@]}"; do
    python run_compile.py --model $2 --gpu $1 --task ${task} &> revision/compile/result_$2_OURS_${task}_vllm_1.txt
done

for task in "${tasks[@]}"; do
    python run_compile.py --model $2 --gpu $1 --task ${task} &> revision/compile/result_$2_OURS_${task}_vllm_2.txt
done

for task in "${tasks[@]}"; do
    python run_compile.py --model $2 --gpu $1 --task ${task} &> revision/compile/result_$2_OURS_${task}_vllm_3.txt
done

for task in "${tasks[@]}"; do
    python run_compile.py --model $2 --gpu $1 --task ${task} &> revision/compile/result_$2_OURS_${task}_vllm_4.txt
done

for task in "${tasks[@]}"; do
    python run_compile.py --model $2 --gpu $1 --task ${task} &> revision/compile/result_$2_OURS_${task}_vllm_5.txt
done
