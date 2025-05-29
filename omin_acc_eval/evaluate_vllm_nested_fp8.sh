#!/bin/bash

# 모델 정의
MODELS=(
    "Llama-3.1-8B"
    "Mistral-Nemo-Base-2407"
    "phi-4"
    "Mistral-Small-24B-Base-2501"
)

# 태스크 정의
tasks=(
    minerva_math
    bbh_zeroshot
    leaderboard_mmlu_pro
)

# 각 모델을 병렬로 실행하는 함수
run_model() {
    local gpu=$1
    local model=$2
    
    echo "Starting model $model on GPU $gpu"
    
    # 각 태스크에 대해 5번 실행
    for run in {1..3}; do
        for task in "${tasks[@]}"; do
            # 주석 처리된 태스크는 건너뛰기
            if [[ $task == \#* ]]; then
                continue
            fi
            
            echo "Running $model on GPU $gpu, task $task, run $run"
            python run_compile.py --model $model --gpu $gpu --task ${task} &> revision/compile/result_${model}_OURS_${task}_vllm_${run}.txt
        done
    done
    
    echo "Completed model $model on GPU $gpu"
}

# 출력 디렉토리 생성
mkdir -p revision/compile

# 각 모델을 백그라운드에서 병렬로 실행
for i in "${!MODELS[@]}"; do
    gpu=$i
    model="${MODELS[$i]}"
    run_model $gpu $model &
done

# 모든 백그라운드 작업이 완료될 때까지 대기
wait

echo "All models completed!"