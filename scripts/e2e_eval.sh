#!/bin/bash
set -e

MODELS=(
  "/home/ubuntu/models/Llama-3.1-8B"
  "/home/ubuntu/models/Mistral-Nemo-Base-2407"
  "/home/ubuntu/models/Mistral-Small-24B-Base-2501"
  "/home/ubuntu/models/phi-4"
)

for MODEL in "${MODELS[@]}"; do
  echo "==== Running FP16 for $MODEL ===="
  python e2e_bench.py --model "$MODEL"

  echo "==== Running NestedFP for $MODEL ===="
  python e2e_bench.py --model "$MODEL" --nestedfp
done
