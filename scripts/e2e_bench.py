#!/usr/bin/env python3
import os
import sys
import csv
import time
import gc
import argparse
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
VLLM_SRC = os.path.abspath(os.path.join(CURRENT_DIR, "..", "vllm"))
RESULT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "results", "e2e"))
os.makedirs(RESULT_DIR, exist_ok=True)

sys.path.insert(0, VLLM_SRC)

from vllm import LLM, SamplingParams


def run_inference_benchmark(llm, output_file, prompts, sampling_params,
                            input_token_size, output_token_size, batch_size):
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    _ = llm.generate(prompts, sampling_params)
    torch.cuda.synchronize()
    latency = time.perf_counter() - start_time

    with open(output_file, mode="a", newline="") as f:
        csv.writer(f).writerow([input_token_size, output_token_size, batch_size, latency])

    print(f"[Input {input_token_size} | Output {output_token_size} | Batch {batch_size}] "
          f"Latency = {latency:.6f} s")


def parse_args():
    p = argparse.ArgumentParser(description="Simple vLLM latency benchmark")
    p.add_argument("--model", required=True, help="Model path")
    p.add_argument("--nestedfp", action="store_true",
                   help="If set, enable nestedfp quantization and add '_nestedfp' tag to outfile.")
    return p.parse_args()


def main():
    args = parse_args()
    model_path = args.model

    base_model_name = os.path.basename(model_path.rstrip("/"))
    quant_tag = "_nestedfp" if args.nestedfp else "_fp16"

    outfile_name = f"{base_model_name}{quant_tag}_latency.csv"
    outfile_path = os.path.join(RESULT_DIR, outfile_name)

    with open(outfile_path, mode="w", newline="") as f:
        csv.writer(f).writerow(["input_token_size", "output_token_size", "batch_size", "latency"])

    input_token_sizes = [16, 256, 1024]
    output_token_sizes = [512]
    batch_sizes = [512, 256, 128, 64, 32]
    CAPTURE_SIZES = [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]

    for batch_size in batch_sizes:
        capture_sizes = [s for s in CAPTURE_SIZES if s <= batch_size]

        llm_kwargs = dict(
            model=model_path,
            dtype="float16",
            max_num_batched_tokens=batch_size,
            max_num_seqs=batch_size,
            enable_prefix_caching=False,
            compilation_config={
                "level": 3,
                "custom_ops": ["none"],
                "splitting_ops": [
                    "vllm.unified_attention",
                    "vllm.unified_attention_with_output",
                ],
                "full_cuda_graph": True,
                "use_inductor": True,
                "compile_sizes": capture_sizes,
                "use_cudagraph": True,
                "cudagraph_num_of_warmups": 2,
                "cudagraph_capture_sizes": capture_sizes,
                "max_capture_size": batch_size,
            },
            gpu_memory_utilization=0.97,
            load_format="dummy",
        )

        if args.nestedfp:
            llm_kwargs["quantization"] = "nestedfp"

        llm = LLM(**llm_kwargs)

        with open(outfile_path, mode="a", newline="") as f:
            csv.writer(f).writerow([f"Batch Size = {batch_size}", "", "", ""])

        for out_tok in output_token_sizes:
            with open(outfile_path, mode="a", newline="") as f:
                csv.writer(f).writerow([f"Output Token Size = {out_tok}", "", "", ""])

            sampling_params = SamplingParams(max_tokens=out_tok, temperature=0, ignore_eos=True)

            for in_tok in input_token_sizes:
                prompt_text = "A" * in_tok
                prompts = [prompt_text] * batch_size
                run_inference_benchmark(
                    llm, outfile_path, prompts, sampling_params,
                    input_token_size=in_tok, output_token_size=out_tok,
                    batch_size=batch_size
                )

        del llm
        torch.cuda.empty_cache()
        gc.collect()

    print(f"Results saved to: {outfile_path}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
