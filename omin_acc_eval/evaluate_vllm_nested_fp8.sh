#!/bin/bash

# 사용할 GPU 목록만 지정 (물리 GPU 번호)
VISIBLE_DEVICES=(0 1 2 3)
TOTAL_GPUS=${#VISIBLE_DEVICES[@]}  # 4로 설정됨
MEMORY_THRESHOLD=500  # MB 단위, GPU 메모리 사용량이 이 값 이하면 idle로 간주


# 설정 변수들 - 여기서 직접 수정하세요
MODELS=(
    "Llama-3.1-8B"
    "Mistral-Nemo-Base-2407"
    "Mistral-Small-24B-Base-2501"
    "phi-4"
)

tasks=(
minerva_math
bbh_zeroshot
# leaderboard_mmlu_pro
)

# Idle GPU 찾기 함수
find_idle_gpus() {
    local idle_gpus=()

    echo "Checking GPU status..." >&2

    for ((i = 0; i < TOTAL_GPUS; i++)); do
        local real_gpu_id=${VISIBLE_DEVICES[$i]}
        memory_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=$real_gpu_id 2>/dev/null)

        if [ $? -eq 0 ] && [ -n "$memory_used" ] && [[ "$memory_used" =~ ^[0-9]+$ ]] && [ "$memory_used" -lt "$MEMORY_THRESHOLD" ]; then
            idle_gpus+=($real_gpu_id)
            echo "GPU $real_gpu_id: IDLE (${memory_used}MB used)" >&2
        elif [ $? -eq 0 ]; then
            echo "GPU $real_gpu_id: BUSY (${memory_used}MB used)" >&2
        fi
    done

    echo "${idle_gpus[@]}"
}

# 특정 모델에 대한 모든 GPU 작업 함수
run_model_evaluation() {
    local model_path=$1
    
    echo "=========================================="
    echo "Starting evaluation for model: $model_path"
    echo "=========================================="
    
    # Idle GPU 목록 가져오기
    IDLE_GPUS=($(find_idle_gpus))
    
    if [ ${#IDLE_GPUS[@]} -eq 0 ]; then
        echo "No idle GPUs found for model $model_path. Skipping..."
        return 1
    fi
    
    echo
    echo "Found ${#IDLE_GPUS[@]} idle GPU(s): ${IDLE_GPUS[@]}"
    echo "Starting evaluation on idle GPUs for model: $model_path"
    echo
    
    # 각 GPU에서 병렬로 실행하는 함수
    run_gpu_tasks() {
        local gpu_id=$1
        local model=$2
        echo "Starting tasks on GPU $gpu_id for model $model"
        
        for task in "${tasks[@]}"; do
            echo "GPU $gpu_id - Model $model - Running $task"
            PYTHONPATH="/disk/vllm:$PYTHONPATH" VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=$gpu_id  HF_ALLOW_CODE_EVAL=1 lm_eval --model vllm --model_args pretrained=/disk/models/$model,tensor_parallel_size=1,add_bos_token=True,dtype="float16",quantization="dualfp",enforce_eager=True --tasks ${task} --batch_size auto --trust_remote_code --confirm_run_unsafe_code &> record/result_nestedfp8_${model}_GPU${gpu_id}_${task}_vllm_1.txt
        done
        
        echo "GPU $gpu_id - Model $model - All tasks completed!"
    }
    
    # 각 idle GPU에서 병렬로 실행
    for gpu_id in "${IDLE_GPUS[@]}"; do
        run_gpu_tasks $gpu_id $model_path &
    done
    
    # 모든 백그라운드 작업이 완료될 때까지 대기
    wait
    
    echo
    echo "Model $model_path evaluation completed!"
    echo
}

# 메인 실행 부분
echo "Multi-Model Evaluation Script"
echo "Models to evaluate: ${MODELS[@]}"
echo "Total GPUs: $TOTAL_GPUS"
echo "Memory threshold: ${MEMORY_THRESHOLD}MB"
echo "Tasks: ${tasks[@]}"
echo

# record 디렉토리 생성
mkdir -p record

# 전체 시작 시간 기록
start_time=$(date)
echo "Evaluation started at: $start_time"
echo

# 각 모델에 대해 순차적으로 실행
for model in "${MODELS[@]}"; do
    model_start_time=$(date)
    echo "Model $model started at: $model_start_time"
    
    run_model_evaluation "$model"
    
    model_end_time=$(date)
    echo "Model $model completed at: $model_end_time"
    echo
    
    # 다음 모델 시작 전 잠시 대기 (GPU 메모리 정리 시간)
    if [ "$model" != "${MODELS[-1]}" ]; then
        echo "Waiting 30 seconds before starting next model..."
        sleep 30
        echo
    fi
done

# 전체 완료 시간 기록
end_time=$(date)
echo "=========================================="
echo "ALL EVALUATIONS COMPLETED!"
echo "Started at: $start_time"
echo "Completed at: $end_time"
echo "Results saved in record/ directory"
echo "=========================================="