import os
import json
import torch
import lm_eval
from tqdm import tqdm

from .dataloader import *

@torch.no_grad()
def evaluate_ppl(model, tokenizer, testcases, chunk_size=2048):
    model.eval()
    results = {}

    for testcase_name in testcases:
        raw_text = get_loaders(testcase_name)
        input_tokens = tokenizer(raw_text, return_tensors='pt')
        input_tokens.to(model.device)
        print(model.device)

        seq_len = input_tokens.input_ids.size(1)
        nsamples = seq_len // chunk_size  # floor(seq_len / chunk_size)

        neg_log_likelihoods = []
        for i in tqdm(range(nsamples)):
            begin_loc = i * chunk_size
            input_ids = input_tokens.input_ids[:, begin_loc:begin_loc + chunk_size]

            # add BOS token for Gemma-7B
            # https://github.com/huggingface/transformers/issues/29250
            if 'gemma' in model.config.architectures[0].lower():
                # Mostly harmless to other models, but a slight drop in ppl is observed
                # Hence, we only add the BOS token for Gemma models for now
                input_ids[:, 0] = tokenizer.bos_token_id

            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                neg_log_likelihood = outputs.loss
                neg_log_likelihoods.append(neg_log_likelihood)

            ppl = torch.exp(torch.stack(neg_log_likelihoods).mean())
            results[f"{testcase_name}"] = ppl.item()

    return results

@torch.no_grad()
def run_lm_eval(model, tokenizer, tasks):
    model.eval()
    results = {}

    model_lm = lm_eval.models.huggingface.HFLM(pretrained=model, tokenizer=tokenizer)
    eval_results = lm_eval.simple_evaluate(model=model_lm, tasks=tasks, confirm_run_unsafe_code=True)

    #for task in tasks:
    #    results[f"{task}"] = eval_results['results'][task]

    return eval_results
