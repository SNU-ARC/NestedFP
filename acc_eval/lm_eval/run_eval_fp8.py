import os
import json
import argparse

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from dual_fp.modules import make_fp8
from dual_fp.evaluate import evaluate_ppl, run_lm_eval

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='fp16')
parser.add_argument('--model', type=str, default='Llama-3.1-8B')
parser.add_argument('--task', type=str, default='gsm8k')
args = parser.parse_args()

model_name = f'/disk/models/{args.model}'
#ppl = ['wikitext2', 'ptb_new_sliced'] # 'c4_new', 
lm_eval = ['arc_easy'] # , 'leaderboard_mmlu_pro' 'minerva_math', 'bbh_zeroshot'
"""
lm_eval = [
    {
        "task": args.task,
        "dataset_kwargs": {
            "trust_remote_code": True
        }
    }
]
"""

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto', device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#if args.mode != 'fp16':
#make_fp8(model, args.mode)

results = {}
#results['ppl'] = evaluate_ppl(model, tokenizer, ppl, chunk_size=2048)
results['lm-eval'] = run_lm_eval(model, tokenizer, lm_eval)
print(results)
