#!/bin/bash

# 사용할 GPU 목록
VISIBLE_DEVICES=(0 1 2 3)
TOTAL_GPUS=${#VISIBLE_DEVICES[@]}
MEMORY_THRESHOLD=500  # MB 기준

# 평가할 모델들
MODELS=(
    "Llama-3.1-8B-FP8-Dynamic-Half"
    "Mistral-Nemo-Base-2407-FP8-Dynamic-Half"
    "Mistral-Small-24B-Base-2501-FP8-Dynamic-Half"
    "phi-4-FP8-Dynamic-Half"
)

# 평가할 작업들
tasks=(
    minerva_math
    bbh_zeroshot
)

# 디코딩 설정 (JSON) - 반드시 큰따옴표로 묶고, 변수는 작은따옴표로 감싸야 함
GEN_KWARGS='{"temperature": 0.0, "do_sample": false}'

# Idle GPU 찾기
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

run_model_evaluation() {
    local model_path=$1

    echo "=========================================="
    echo "Starting evaluation for model: $model_path"
    echo "=========================================="

    IDLE_GPUS=($(find_idle_gpus))

    if [ ${#IDLE_GPUS[@]} -eq 0 ]; then
        echo "No idle GPUs found for model $model_path. Skipping..."
        return 1
    fi

    echo
    echo "Found ${#IDLE_GPUS[@]} idle GPU(s): ${IDLE_GPUS[@]}"
    echo

    run_gpu_tasks() {
        local gpu_id=$1
        local model=$2

        echo "Starting tasks on GPU $gpu_id for model $model"

        for task in "${tasks[@]}"; do
            echo "GPU $gpu_id - Model $model - Running $task"

            # 환경변수 포함 평가 실행
            PYTHONPATH="/disk/vllm:$PYTHONPATH" \
            VLLM_WORKER_MULTIPROC_METHOD=spawn \
            CUDA_VISIBLE_DEVICES=$gpu_id \
            PYTHONHASHSEED=0 \
            CUBLAS_WORKSPACE_CONFIG=:4096:8 \
            CUDA_LAUNCH_BLOCKING=1 \
            TORCH_USE_DETERMINISTIC_ALGORITHMS=1 \
            HF_ALLOW_CODE_EVAL=1 \
            lm_eval --model vllm \
              --model_args pretrained=/disk/models/$model,tensor_parallel_size=1,add_bos_token=True,dtype="float16",enforce_eager=True \
              --gen_kwargs="$GEN_KWARGS" \
              --tasks ${task} \
              --batch_size auto \
              --trust_remote_code \
              --confirm_run_unsafe_code \
              --verbosity WARNING \
              &> record/result_fp8_${model}_GPU${gpu_id}_${task}_vllm_deterministic.txt
        done

        echo "GPU $gpu_id - Model $model - All tasks completed!"
    }

    for gpu_id in "${IDLE_GPUS[@]}"; do
        run_gpu_tasks $gpu_id $model_path &
    done

    wait

    echo
    echo "Model $model_path evaluation completed!"
    echo
}

# 메인 실행
echo "Multi-Model Evaluation Script (Deterministic Mode)"
echo "Models to evaluate: ${MODELS[@]}"
echo "Total visible GPUs: $TOTAL_GPUS"
echo "GPUs to use: ${VISIBLE_DEVICES[@]}"
echo "Memory threshold: ${MEMORY_THRESHOLD}MB"
echo "Tasks: ${tasks[@]}"
echo

mkdir -p record

start_time=$(date)
echo "Evaluation started at: $start_time"
echo

for model in "${MODELS[@]}"; do
    model_start_time=$(date)
    echo "Model $model started at: $model_start_time"

    run_model_evaluation "$model"

    model_end_time=$(date)
    echo "Model $model completed at: $model_end_time"
    echo

    if [ "$model" != "${MODELS[-1]}" ]; then
        echo "Waiting 30 seconds before starting next model..."
        sleep 30
        echo
    fi
done

end_time=$(date)
echo "=========================================="
echo "ALL EVALUATIONS COMPLETED!"
echo "Started at: $start_time"
echo "Completed at: $end_time"
echo "Results saved in record/ directory"
echo "=========================================="
