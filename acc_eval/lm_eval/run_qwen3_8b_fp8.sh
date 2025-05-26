#!/bin/bash

lm_eval --model hf \
    --model_args pretrained=/disk/models/Qwen3-8B-FP8,device_map=cuda:0 \
    --tasks gsm8k \
    --device cuda:0 \
    --batch_size 8 \
    &> result_qwen3_8b_fp8.txt
