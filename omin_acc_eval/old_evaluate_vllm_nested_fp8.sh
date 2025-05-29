#!/bin/bash

# 사용할 GPU 목록만 지정 (물리 GPU 번호)
VISIBLE_DEVICES=(0 1 2 3)
TOTAL_GPUS=${#VISIBLE_DEVICES[@]}
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

# 반복 실행 횟수 설정
REPEAT_COUNT=5  # 각 태스크를 몇 번 반복할지 설정

# Idle GPU 찾기 함수
find_idle_gpus() {
    local idle_gpus=()

    echo "Checking GPU status..." >&2
    
    for gpu_id in "${VISIBLE_DEVICES[@]}"; do
        # nvidia-smi를 사용하여 GPU 메모리 사용량 확인
        memory_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=$gpu_id 2>/dev/null)
        
        if [ $? -eq 0 ] && [ "$memory_used" -le "$MEMORY_THRESHOLD" ]; then
            idle_gpus+=($gpu_id)
            echo "GPU $gpu_id is idle (${memory_used}MB used)" >&2
        else
            echo "GPU $gpu_id is busy (${memory_used}MB used)" >&2
        fi
    done

    echo "${idle_gpus[@]}"
}

# 특정 GPU에서 특정 모델의 모든 태스크를 반복 실행하는 함수
run_gpu_model_tasks() {
    local gpu_id=$1
    local model=$2
    
    echo "=========================================="
    echo "GPU $gpu_id - Starting evaluation for model: $model"
    echo "=========================================="
    
    for task in "${tasks[@]}"; do
        echo "GPU $gpu_id - Model $model - Starting task: $task"
        echo "Will run $REPEAT_COUNT times"
        
        for ((run=1; run<=REPEAT_COUNT; run++)); do
            echo "GPU $gpu_id - Model $model - Task $task - Run $run/$REPEAT_COUNT"
            
            CUDA_VISIBLE_DEVICES=$gpu_id \
            PYTHONPATH="/disk/revision/vllm:$PYTHONPATH" \
            VLLM_USE_V1=1 \
            VLLM_FLASH_ATTN_VERSION=3 \
            VLLM_WORKER_MULTIPROC_METHOD=spawn \
            HF_ALLOW_CODE_EVAL=1 \
            lm_eval \
                --model vllm \
                --model_args pretrained=/disk/models/$model,tensor_parallel_size=1,add_bos_token=True,dtype=float16,quantization=nestedfp \
                --tasks ${task} \
                --batch_size auto \
                --trust_remote_code \
                --confirm_run_unsafe_code \
                &> record/result_nestedfp8_${model}_GPU${gpu_id}_${task}_run${run}_vllm_1.txt
            
            echo "GPU $gpu_id - Model $model - Task $task - Run $run completed"
            
            # 실행 간 잠시 대기 (메모리 정리)
            if [ $run -lt $REPEAT_COUNT ]; then
                sleep 10
            fi
        done
        
        echo "GPU $gpu_id - Model $model - Task $task - All $REPEAT_COUNT runs completed!"
    done
    
    echo "GPU $gpu_id - Model $model - All tasks completed!"
}

# 메인 실행 부분
echo "Multi-GPU Multi-Model Evaluation Script"
echo "Models to evaluate: ${MODELS[@]}"
echo "Total GPUs: $TOTAL_GPUS"
echo "Memory threshold: ${MEMORY_THRESHOLD}MB"
echo "Tasks: ${tasks[@]}"
echo "Repeat count per task: $REPEAT_COUNT"
echo

# 모델 수와 GPU 수 확인
if [ ${#MODELS[@]} -ne $TOTAL_GPUS ]; then
    echo "Warning: Number of models (${#MODELS[@]}) doesn't match number of GPUs ($TOTAL_GPUS)"
    echo "Each GPU will be assigned to models in order, some may be skipped or repeated"
fi

# record 디렉토리 생성
mkdir -p record

# 스크립트 시작 시 한 번만 vLLM 캐시 정리
# echo "Clearing torch compile cache..."
# rm -rf /home/ubuntu/.cache/vllm/torch_compile_cache/
# echo "Cache cleared"
# echo

# Idle GPU 목록 가져오기
IDLE_GPUS=($(find_idle_gpus))

if [ ${#IDLE_GPUS[@]} -eq 0 ]; then
    echo "No idle GPUs found. Exiting..."
    exit 1
fi

echo "Found ${#IDLE_GPUS[@]} idle GPU(s): ${IDLE_GPUS[@]}"

# 사용할 GPU 수를 idle GPU 수와 모델 수 중 작은 값으로 설정
USE_GPUS=${#IDLE_GPUS[@]}
if [ ${#MODELS[@]} -lt $USE_GPUS ]; then
    USE_GPUS=${#MODELS[@]}
fi

echo "Will use $USE_GPUS GPU(s) for evaluation"
echo

# 전체 시작 시간 기록
start_time=$(date)
echo "Evaluation started at: $start_time"
echo

# 각 GPU에 모델을 할당하여 병렬 실행
for ((i=0; i<USE_GPUS; i++)); do
    gpu_id=${IDLE_GPUS[$i]}
    model=${MODELS[$i]}
    
    echo "Assigning GPU $gpu_id to model $model"
    run_gpu_model_tasks $gpu_id $model &
done

echo
echo "All GPU tasks started. Waiting for completion..."
echo

# 모든 백그라운드 작업이 완료될 때까지 대기
wait

# 전체 완료 시간 기록
end_time=$(date)
echo
echo "=========================================="
echo "ALL EVALUATIONS COMPLETED!"
echo "Started at: $start_time"
echo "Completed at: $end_time"
echo "Results saved in record/ directory"
echo "Each task was repeated $REPEAT_COUNT times per model"
echo "=========================================="