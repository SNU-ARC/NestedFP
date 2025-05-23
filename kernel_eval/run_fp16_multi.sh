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


# # (2048, 10240)
# # (2048, 8192)
# # (2048, 57344)
# # (7168, 8192)

# N_VALUES=(2048 2048 2048 7168)
# K_VALUES=(10240 8192 57344 8192)


# 총 8개 GPU 사용
GPU_IDS=(0 1 2 3 4 5 6 7)

# M 범위를 나눔
M_RANGES=("32 1024" "1056 2048")

# 작업 리스트
JOBS=()
JOB_COUNTER=0

# 각 (N,K) × M 범위 조합을 작업으로 등록
for i in "${!N_VALUES[@]}"; do
  for m_range in "${M_RANGES[@]}"; do
    read -r M_START M_END <<< "$m_range"
    
    GPU="${GPU_IDS[$((JOB_COUNTER % ${#GPU_IDS[@]}))]}"
    
    echo "작업 배치: N=${N_VALUES[$i]}, K=${K_VALUES[$i]}, GPU=$GPU, M=$M_START~$M_END"
    
    JOBS+=("./run_fp16_single.sh ${N_VALUES[$i]} ${K_VALUES[$i]} $GPU $M_START $M_END")
    
    JOB_COUNTER=$((JOB_COUNTER + 1))
  done
done

# 병렬 실행
for job in "${JOBS[@]}"; do
  echo "실행: $job"
  eval "$job" &
done

wait
echo "== FP16 커널 실험 전체 완료 =="
