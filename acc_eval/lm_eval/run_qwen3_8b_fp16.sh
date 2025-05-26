#!/bin/bash

lm_eval --model hf \
    --model_args pretrained=/disk/models/Qwen3-8B,dtype="float16" \
    --tasks gsm8k \
    --device cuda:1 \
    --batch_size 8 \
    &> result_qwen3_8b_fp16.txt
