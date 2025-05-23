#!/bin/bash

N="$1"
K="$2"
GPU="$3"
M_START="$4"
M_END="$5"

CALL_DIR=$(pwd)
SCRIPT_DIR=$(pwd)



# === Kernel 목록 (36 + 22 = 총 58개) ===
baseline_no_coop_kernels=(
  cutlass_tma_warp_specialized_64_16_64 # 1
  cutlass_tma_warp_specialized_64_16_128 # 2
  cutlass_tma_warp_specialized_64_16_256 # 3
  cutlass_tma_warp_specialized_64_32_64 # 4
  cutlass_tma_warp_specialized_64_32_128 # 6 
  cutlass_tma_warp_specialized_64_32_256 # 7 
  cutlass_tma_warp_specialized_64_64_64 # 9 
  cutlass_tma_warp_specialized_64_64_128 # 10
  cutlass_tma_warp_specialized_64_64_256 # 11
  cutlass_tma_warp_specialized_64_128_64  # 12
  cutlass_tma_warp_specialized_64_128_128 # 13
  cutlass_tma_warp_specialized_64_128_256
  cutlass_tma_warp_specialized_64_256_64
  cutlass_tma_warp_specialized_64_256_128
  cutlass_tma_warp_specialized_128_16_64
  cutlass_tma_warp_specialized_128_16_128
  cutlass_tma_warp_specialized_128_16_256
  cutlass_tma_warp_specialized_128_32_64
  cutlass_tma_warp_specialized_128_32_128
  cutlass_tma_warp_specialized_128_32_256
  cutlass_tma_warp_specialized_128_64_64
  cutlass_tma_warp_specialized_128_64_128
  cutlass_tma_warp_specialized_128_64_256
  cutlass_tma_warp_specialized_128_128_64
  cutlass_tma_warp_specialized_128_128_128
  cutlass_tma_warp_specialized_128_256_64
  cutlass_tma_warp_specialized_128_256_128
  cutlass_tma_warp_specialized_256_16_64
  cutlass_tma_warp_specialized_256_16_128
  cutlass_tma_warp_specialized_256_32_64
  cutlass_tma_warp_specialized_256_32_128
  cutlass_tma_warp_specialized_256_64_64
  cutlass_tma_warp_specialized_256_64_128
  cutlass_tma_warp_specialized_256_128_64
  cutlass_tma_warp_specialized_256_128_128
  cutlass_tma_warp_specialized_256_256_64
)
baseline_coop_kernels=(
  # #cluster shape : <2, 1, 1>
  cutlass_tma_warp_specialized_cooperative_2_1_1_128_16_64
  cutlass_tma_warp_specialized_cooperative_2_1_1_128_16_128
  cutlass_tma_warp_specialized_cooperative_2_1_1_128_16_256
  cutlass_tma_warp_specialized_cooperative_2_1_1_128_32_64
  cutlass_tma_warp_specialized_cooperative_2_1_1_128_32_128
  cutlass_tma_warp_specialized_cooperative_2_1_1_128_32_256
  cutlass_tma_warp_specialized_cooperative_2_1_1_128_64_64
  cutlass_tma_warp_specialized_cooperative_2_1_1_128_64_128
  cutlass_tma_warp_specialized_cooperative_2_1_1_128_64_256
  cutlass_tma_warp_specialized_cooperative_2_1_1_128_128_64
  cutlass_tma_warp_specialized_cooperative_2_1_1_128_128_128
  cutlass_tma_warp_specialized_cooperative_2_1_1_128_256_64
  cutlass_tma_warp_specialized_cooperative_2_1_1_128_256_128
  cutlass_tma_warp_specialized_cooperative_2_1_1_256_16_64
  cutlass_tma_warp_specialized_cooperative_2_1_1_256_16_128
  cutlass_tma_warp_specialized_cooperative_2_1_1_256_32_64
  cutlass_tma_warp_specialized_cooperative_2_1_1_256_32_128
  cutlass_tma_warp_specialized_cooperative_2_1_1_256_64_64
  cutlass_tma_warp_specialized_cooperative_2_1_1_256_64_128
  cutlass_tma_warp_specialized_cooperative_2_1_1_256_128_64
  cutlass_tma_warp_specialized_cooperative_2_1_1_256_128_128
  cutlass_tma_warp_specialized_cooperative_2_1_1_256_256_64
)


baseline_streamk_kernels=(
  cutlass_tma_warp_specialized_cooperative_streamk_2_1_1_128_16_64
  cutlass_tma_warp_specialized_cooperative_streamk_2_1_1_128_16_128
  cutlass_tma_warp_specialized_cooperative_streamk_2_1_1_128_16_256
  cutlass_tma_warp_specialized_cooperative_streamk_2_1_1_128_32_64
  cutlass_tma_warp_specialized_cooperative_streamk_2_1_1_128_32_128
  cutlass_tma_warp_specialized_cooperative_streamk_2_1_1_128_32_256
  cutlass_tma_warp_specialized_cooperative_streamk_2_1_1_128_64_64
  cutlass_tma_warp_specialized_cooperative_streamk_2_1_1_128_64_128
  cutlass_tma_warp_specialized_cooperative_streamk_2_1_1_128_64_256
  cutlass_tma_warp_specialized_cooperative_streamk_2_1_1_128_128_64
  cutlass_tma_warp_specialized_cooperative_streamk_2_1_1_128_128_128
  cutlass_tma_warp_specialized_cooperative_streamk_2_1_1_128_256_64
  cutlass_tma_warp_specialized_cooperative_streamk_2_1_1_128_256_128
  cutlass_tma_warp_specialized_cooperative_streamk_2_1_1_256_16_64
  cutlass_tma_warp_specialized_cooperative_streamk_2_1_1_256_16_128
  cutlass_tma_warp_specialized_cooperative_streamk_2_1_1_256_32_64
  cutlass_tma_warp_specialized_cooperative_streamk_2_1_1_256_32_128
  cutlass_tma_warp_specialized_cooperative_streamk_2_1_1_256_64_64
  cutlass_tma_warp_specialized_cooperative_streamk_2_1_1_256_64_128
  cutlass_tma_warp_specialized_cooperative_streamk_2_1_1_256_128_64
  cutlass_tma_warp_specialized_cooperative_streamk_2_1_1_256_128_128
  cutlass_tma_warp_specialized_cooperative_streamk_2_1_1_256_256_64
)

custom_streamk_kernels=(
  cutlass_tma_warp_specialized_cooperative_streamk_custom_2_1_1_128_16_64
  cutlass_tma_warp_specialized_cooperative_streamk_custom_2_1_1_128_16_128
  cutlass_tma_warp_specialized_cooperative_streamk_custom_2_1_1_128_16_256
  cutlass_tma_warp_specialized_cooperative_streamk_custom_2_1_1_128_32_64
  cutlass_tma_warp_specialized_cooperative_streamk_custom_2_1_1_128_32_128
  cutlass_tma_warp_specialized_cooperative_streamk_custom_2_1_1_128_32_256
  cutlass_tma_warp_specialized_cooperative_streamk_custom_2_1_1_128_64_64
  cutlass_tma_warp_specialized_cooperative_streamk_custom_2_1_1_128_64_128
  cutlass_tma_warp_specialized_cooperative_streamk_custom_2_1_1_128_64_256
  cutlass_tma_warp_specialized_cooperative_streamk_custom_2_1_1_128_128_64
  cutlass_tma_warp_specialized_cooperative_streamk_custom_2_1_1_128_128_128
  cutlass_tma_warp_specialized_cooperative_streamk_custom_2_1_1_128_256_64
  cutlass_tma_warp_specialized_cooperative_streamk_custom_2_1_1_128_256_128
  cutlass_tma_warp_specialized_cooperative_streamk_custom_2_1_1_256_16_64
  cutlass_tma_warp_specialized_cooperative_streamk_custom_2_1_1_256_16_128
  cutlass_tma_warp_specialized_cooperative_streamk_custom_2_1_1_256_32_64
  cutlass_tma_warp_specialized_cooperative_streamk_custom_2_1_1_256_32_128
  cutlass_tma_warp_specialized_cooperative_streamk_custom_2_1_1_256_64_64
  cutlass_tma_warp_specialized_cooperative_streamk_custom_2_1_1_256_64_128
  cutlass_tma_warp_specialized_cooperative_streamk_custom_2_1_1_256_128_64
  cutlass_tma_warp_specialized_cooperative_streamk_custom_2_1_1_256_128_128
  cutlass_tma_warp_specialized_cooperative_streamk_custom_2_1_1_256_256_64
)


custom_no_coop_kernels=()
for k in "${baseline_no_coop_kernels[@]}"; do
  custom_no_coop_kernels+=("${k/_specialized/_specialized_custom}")
done

custom_coop_kernels=()
for k in "${baseline_coop_kernels[@]}"; do
  if [[ "$k" =~ ^(cutlass_tma_warp_specialized_cooperative_([0-9]+)_([0-9]+)_([0-9]+))_(.*)$ ]]; then
    prefix="${BASH_REMATCH[1]}"
    suffix="${BASH_REMATCH[5]}"
    custom="${prefix}_custom_${suffix}"
    custom_coop_kernels+=("$custom")
  else
    echo "⚠️ 경고: 커널 이름 형식이 예상과 다름 → $k"
  fi
done



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
OUT_BASELINE="${OUT_PREFIX}_baseline.csv"
OUT_CUSTOM="${OUT_PREFIX}_custom.csv"

CUSTOM_KERNEL_COUNT=$((${#custom_no_coop_kernels[@]} + ${#custom_coop_kernels[@]} + ${#custom_streamk_kernels[@]}))
BASELINE_KERNEL_COUNT=$((${#baseline_no_coop_kernels[@]} + ${#baseline_coop_kernels[@]} + ${#baseline_streamk_kernels[@]}))

write_header "$OUT_CUSTOM" "custom" "$CUSTOM_KERNEL_COUNT"
write_header "$OUT_BASELINE" "baseline" "$BASELINE_KERNEL_COUNT"

for ((m=$M_START; m<=$M_END; m+=32)); do
  echo -n "$m" >> "$OUT_CUSTOM"
  for func in "${custom_no_coop_kernels[@]}" "${custom_coop_kernels[@]}" "${custom_streamk_kernels[@]}"; do
      TMP=$(mktemp)
      CUDA_VISIBLE_DEVICES="$GPU" ncu --kernel-name "device_kernel" --metrics gpu__time_duration.sum \
        --cache-control all --clock-control none \
        python ncu.py --m="$m" --n="$N" --k="$K" --func="$func" > "$TMP" 2>&1

      if grep -q "==ERROR==" "$TMP"; then
        LATENCY=""
      else
        LATENCY=$(get_latency "$TMP")
      fi
      [[ -z "$LATENCY" ]] && echo "[$m] $func → LATENCY not found" >> "$CALL_DIR/latency_error.log"
      rm -f "$TMP"
      [[ -z "$LATENCY" ]] && echo -n "," || echo -n ",$LATENCY"
  done >> "$OUT_CUSTOM"
  echo >> "$OUT_CUSTOM"

  echo -n "$m" >> "$OUT_BASELINE"
  for func in "${baseline_no_coop_kernels[@]}" "${baseline_coop_kernels[@]}"  "${baseline_streamk_kernels[@]}" ; do
      TMP=$(mktemp)
      CUDA_VISIBLE_DEVICES="$GPU" ncu --kernel-name "device_kernel" --metrics gpu__time_duration.sum \
        --cache-control all --clock-control none \
        python ncu.py --m="$m" --n="$N" --k="$K" --func="$func" > "$TMP" 2>&1

      if grep -q "==ERROR==" "$TMP"; then
        LATENCY=""
      else
        LATENCY=$(get_latency "$TMP")
      fi
      [[ -z "$LATENCY" ]] && echo "[$m] $func → LATENCY not found" >> "$CALL_DIR/latency_error.log"
      rm -f "$TMP"
      [[ -z "$LATENCY" ]] && echo -n "," || echo -n ",$LATENCY"
  done >> "$OUT_BASELINE"
  echo >> "$OUT_BASELINE"
done

echo "GPU $GPU → 저장 완료: $OUT_BASELINE, $OUT_CUSTOM" >&2