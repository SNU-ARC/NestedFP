#!/bin/bash
# Usage: bash scripts/acc_eval.sh 0,1,2,3 Mistral-Small-24B-Base-2501 bbh_zeroshot
#        bash scripts/acc_eval.sh 0 Mistral-Small-24B-Base-2501 bbh_zeroshot

GPUS="${1:-0}"
MODEL=${2:-Mistral-Small-24B-Base-2501}
TASK=${3:-bbh_zeroshot}

OLD_IFS="$IFS"
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
IFS="$OLD_IFS"
NUM_GPUS=${#GPU_ARRAY[@]}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTDIR="${SCRIPT_DIR}/../results/acc_eval"
mkdir -p "$OUTDIR"

MODEL_NAME=$(basename "$MODEL")
OUTFILE="${OUTDIR}/${MODEL_NAME}_${TASK}_${NUM_GPUS}_GPU_NestedFP.log"

echo "[INFO] Running ACC eval (NestedFP) on GPUs ${GPUS} (${NUM_GPUS} total)"
echo "[INFO] Model: ${MODEL} | Task: ${TASK}"
echo "[INFO] Logging to ${OUTFILE}"

PYTHONPATH="$(pwd)/vllm:$PYTHONPATH" \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
CUDA_VISIBLE_DEVICES=$GPUS \
VLLM_USE_V1=1 \
VLLM_FLASH_ATTN_VERSION=3 \
HF_ALLOW_CODE_EVAL=1 \
lm_eval \
  --model vllm \
  --model_args pretrained=${MODEL},tensor_parallel_size=${NUM_GPUS},add_bos_token=True,dtype="float16",quantization="dualfp" \
  --tasks ${TASK} \
  --batch_size auto \
  --trust_remote_code \
  --confirm_run_unsafe_code \
  &> "$OUTFILE"

echo "[DONE] Output saved to $OUTFILE"
