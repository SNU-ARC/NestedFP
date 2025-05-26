#!/bin/bash

lm_eval --model hf \
    --model_args pretrained=/disk/models/Qwen3-14B,dtype="float16" \
    --tasks gsm8k \
    --device cuda:1 \
    --batch_size 8 \
    &> result_qwen3_14b_fp16.txt
