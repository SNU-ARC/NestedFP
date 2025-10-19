#!/bin/bash

# # Llama 3.1 8B
# N_VALUES=(4096 6144 28672 4096)
# K_VALUES=(4096 4096 4096 14336)


# Mistral-Nemo-Base
# N_VALUES=(5120 28672 5120 6144)
# K_VALUES=(4096 5120 14336 5120)



# # Mistral-Small-24B
# N_VALUES=(6144 5120 65536 5120)
# K_VALUES=(5120 4096 5120 32768)

# Phi-4
N_VALUES=(5120 7680 35840 5120)
K_VALUES=(5120 5120 5120 17920)


# GPU 배정
GPU_IDS=(4 5 6 7)


M_RANGES=("32 2048")

# 작업 설정
JOBS=()
JOB_COUNTER=0

# (N, K) 조합과 M 범위를 조합하여 작업 배치
for i in "${!N_VALUES[@]}"; do
  for m_range in "${M_RANGES[@]}"; do
    read -r M_START M_END <<< "$m_range"
    
    GPU="${GPU_IDS[$JOB_COUNTER % ${#GPU_IDS[@]}]}"
    
    echo "작업 배치: N=${N_VALUES[$i]}, K=${K_VALUES[$i]}, GPU=$GPU, M_범위: $M_START-$M_END"
    
    JOBS+=("./run_fp8_single.sh ${N_VALUES[$i]} ${K_VALUES[$i]} $GPU $M_START $M_END")
    
    JOB_COUNTER=$((JOB_COUNTER + 1))
  done
done

# 병렬로 모든 작업 실행
for job in "${JOBS[@]}"; do
  echo "실행: $job"
  eval "$job" &
done

# 모든 작업이 완료될 때까지 대기
wait

echo "== 모든 FP8 커널 테스트 완료 =="