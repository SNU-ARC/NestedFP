#!/bin/bash

N="$1"
K="$2"
GPU="$3"
M_START="$4"
M_END="$5"

CALL_DIR=$(pwd)
SCRIPT_DIR=$(pwd)

# === FP8 Kernel 목록 ===
fp8_kernels=(
  # 64_*_* 커널들 - 비협력적(no-cooperative)
  cutlass_tma_warp_specialized_fp8_scale_2_1_1_64_16_128
  cutlass_tma_warp_specialized_fp8_scale_2_1_1_64_16_256
  cutlass_tma_warp_specialized_fp8_scale_2_1_1_64_16_512
  cutlass_tma_warp_specialized_fp8_scale_2_1_1_64_32_128
  cutlass_tma_warp_specialized_fp8_scale_2_1_1_64_32_256
  cutlass_tma_warp_specialized_fp8_scale_2_1_1_64_32_512
  cutlass_tma_warp_specialized_fp8_scale_2_1_1_64_64_128
  cutlass_tma_warp_specialized_fp8_scale_2_1_1_64_64_256
  cutlass_tma_warp_specialized_fp8_scale_2_1_1_64_64_512
  cutlass_tma_warp_specialized_fp8_scale_2_1_1_64_128_128
  cutlass_tma_warp_specialized_fp8_scale_2_1_1_64_128_256
  cutlass_tma_warp_specialized_fp8_scale_2_1_1_64_128_512
  cutlass_tma_warp_specialized_fp8_scale_2_1_1_64_256_128
  cutlass_tma_warp_specialized_fp8_scale_2_1_1_64_256_256
  
  # 128_*_* 커널들 - 비협력적(no-cooperative)
  cutlass_tma_warp_specialized_fp8_scale_2_1_1_128_16_128
  cutlass_tma_warp_specialized_fp8_scale_2_1_1_128_16_256
  cutlass_tma_warp_specialized_fp8_scale_2_1_1_128_16_512
  cutlass_tma_warp_specialized_fp8_scale_2_1_1_128_32_128
  cutlass_tma_warp_specialized_fp8_scale_2_1_1_128_32_256
  cutlass_tma_warp_specialized_fp8_scale_2_1_1_128_32_512
  cutlass_tma_warp_specialized_fp8_scale_2_1_1_128_64_128
  cutlass_tma_warp_specialized_fp8_scale_2_1_1_128_64_256
  cutlass_tma_warp_specialized_fp8_scale_2_1_1_128_64_512
  cutlass_tma_warp_specialized_fp8_scale_2_1_1_128_128_128
  cutlass_tma_warp_specialized_fp8_scale_2_1_1_128_128_256
  cutlass_tma_warp_specialized_fp8_scale_2_1_1_128_256_128
  cutlass_tma_warp_specialized_fp8_scale_2_1_1_128_256_256
  
  # 256_*_* 커널들 - 비협력적(no-cooperative)
  cutlass_tma_warp_specialized_fp8_scale_2_1_1_256_16_128
  cutlass_tma_warp_specialized_fp8_scale_2_1_1_256_16_256
  cutlass_tma_warp_specialized_fp8_scale_2_1_1_256_32_128
  cutlass_tma_warp_specialized_fp8_scale_2_1_1_256_32_256
  cutlass_tma_warp_specialized_fp8_scale_2_1_1_256_64_128
  cutlass_tma_warp_specialized_fp8_scale_2_1_1_256_64_256
  cutlass_tma_warp_specialized_fp8_scale_2_1_1_256_128_128
  cutlass_tma_warp_specialized_fp8_scale_2_1_1_256_128_256
  cutlass_tma_warp_specialized_fp8_scale_2_1_1_256_256_128

  # 128_*_* 협력적(cooperative) 커널들
  cutlass_tma_warp_specialized_cooperative_fp8_scale_2_1_1_128_16_128
  cutlass_tma_warp_specialized_cooperative_fp8_scale_2_1_1_128_16_256
  cutlass_tma_warp_specialized_cooperative_fp8_scale_2_1_1_128_16_512
  cutlass_tma_warp_specialized_cooperative_fp8_scale_2_1_1_128_32_128
  cutlass_tma_warp_specialized_cooperative_fp8_scale_2_1_1_128_32_256
  cutlass_tma_warp_specialized_cooperative_fp8_scale_2_1_1_128_32_512
  cutlass_tma_warp_specialized_cooperative_fp8_scale_2_1_1_128_64_128
  cutlass_tma_warp_specialized_cooperative_fp8_scale_2_1_1_128_64_256
  cutlass_tma_warp_specialized_cooperative_fp8_scale_2_1_1_128_64_512
  cutlass_tma_warp_specialized_cooperative_fp8_scale_2_1_1_128_128_128
  cutlass_tma_warp_specialized_cooperative_fp8_scale_2_1_1_128_128_256
  cutlass_tma_warp_specialized_cooperative_fp8_scale_2_1_1_128_256_128
  cutlass_tma_warp_specialized_cooperative_fp8_scale_2_1_1_128_256_256
  
  # 256_*_* 협력적(cooperative) 커널들
  cutlass_tma_warp_specialized_cooperative_fp8_scale_2_1_1_256_16_128
  cutlass_tma_warp_specialized_cooperative_fp8_scale_2_1_1_256_16_256
  cutlass_tma_warp_specialized_cooperative_fp8_scale_2_1_1_256_32_128
  cutlass_tma_warp_specialized_cooperative_fp8_scale_2_1_1_256_32_256
  cutlass_tma_warp_specialized_cooperative_fp8_scale_2_1_1_256_64_128
  cutlass_tma_warp_specialized_cooperative_fp8_scale_2_1_1_256_64_256
  cutlass_tma_warp_specialized_cooperative_fp8_scale_2_1_1_256_128_128
  cutlass_tma_warp_specialized_cooperative_fp8_scale_2_1_1_256_128_256
  cutlass_tma_warp_specialized_cooperative_fp8_scale_2_1_1_256_256_128
)

write_header() {
  local outfile="$1"
  local prefix="$2"
  local count="$3"
  header="M"
  for i in $(seq 1 $count); do
    header+=",${prefix}_$i"
  done
  echo "$header" > "$outfile"
}

get_latency() {
  local file="$1"
  # 더 유연한 패턴 매칭으로 수정
  awk '
    /gpu__time_duration\.sum/ && ($2 == "us" || $2 == "usecond") { print $3; exit }
    /gpu__time_duration\.sum/ && ($2 == "ms" || $2 == "msecond") { printf("%.2f\n", $3 * 1000); exit }
  ' "$file"
}

OUT_PREFIX="$CALL_DIR/n${N}_k${K}_m${M_START}_${M_END}"
OUT_FP8="${OUT_PREFIX}_fp8.csv"

FP8_KERNEL_COUNT=${#fp8_kernels[@]}

write_header "$OUT_FP8" "fp8" "$FP8_KERNEL_COUNT"

for m in $(seq $M_START 32 $M_END); do
  echo "Processing M=$m..." >&2
  
  # 모든 FP8 커널 처리 (no-coop 및 coop 통합)
  echo -n "$m" >> "$OUT_FP8"
  for func in "${fp8_kernels[@]}"; do
      echo "  Testing $func..." >&2
      TMP=$(mktemp)
      # 디버깅을 위해 전체 출력 저장
      CUDA_VISIBLE_DEVICES="$GPU" ncu --kernel-name "device_kernel" --metrics gpu__time_duration.sum \
        --cache-control all --clock-control none \
        python ncu.py --m="$m" --n="$N" --k="$K" --func="$func" > "$TMP" 2>&1

      # TMP 파일의 내용을 로그 파일에 추가 (디버깅 목적)
      if grep -q "==ERROR==" "$TMP" || ! grep -q "gpu__time_duration.sum" "$TMP"; then
        echo "[$m] $func → 오류 발생 또는 측정값 없음" >> "$CALL_DIR/fp8_latency_debug.log"
        echo "====== 전체 출력 ======" >> "$CALL_DIR/fp8_latency_debug.log"
        cat "$TMP" >> "$CALL_DIR/fp8_latency_debug.log"
        echo "=======================" >> "$CALL_DIR/fp8_latency_debug.log"
        LATENCY=""
      else
        LATENCY=$(get_latency "$TMP")
        if [[ -z "$LATENCY" ]]; then
          echo "[$m] $func → LATENCY 파싱 실패" >> "$CALL_DIR/fp8_latency_error.log"
        fi
      fi
      rm -f "$TMP"
      [[ -z "$LATENCY" ]] && echo -n "," || echo -n ",$LATENCY"
  done >> "$OUT_FP8"
  echo >> "$OUT_FP8"
done

echo "GPU $GPU → 저장 완료: $OUT_FP8" >&2