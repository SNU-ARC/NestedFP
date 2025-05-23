import os
import subprocess
import torch
import time
import csv
import gc 


from vllm import LLM, SamplingParams

# 경로 설정
# sys.path.append("/disk/")

import sys
sys.path.insert(0, "/disk/vllm")  # vllm 소스 디렉토리의 상위 경로
import common_globals

def get_idle_gpu():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    gpu_usage = result.stdout.strip().split("\n")
    gpu_usage = [int(line.split(",")[1].strip()) for line in gpu_usage]
    idle_gpu = gpu_usage.index(min(gpu_usage))
    return idle_gpu

def run_inference_benchmark(llm, output_file, prompts, sampling_params, input_token_size, output_token_size, batch_size, verbose=False):
    '''
    주어진 llm 인스턴스에 대해 inference latency를 측정하고 기록
    '''
    common_globals.global_llama_forward_time_ms = 0

    torch.cuda.synchronize()
    start_time = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    torch.cuda.synchronize()
    end_time = time.perf_counter()

    latency = end_time - start_time

    # CSV에 기록
    with open(output_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([input_token_size, output_token_size, batch_size, latency])

    if verbose:
        print(f"[Input {input_token_size} tokens | Output {output_token_size} tokens | Batch {batch_size}] Latency = {latency:.6f} sec")

def main():
    # Set CUDA_VISIBLE_DEVICES to the idle GPU
    idle_gpu = get_idle_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(idle_gpu)
    print(f"Using GPU {idle_gpu}")
    
    model_path = "/disk/models/Mistral-Nemo-Base-2407-FP8-Dynamic"
    output_file_fp8_baseline = "mistralnemo_fp8_baseline.csv"
   
    
    
    # input token size : 32, 256, 1024
    input_token_sizes = [32, 1024] 
    output_token_sizes = [512, 32]
    batch_sizes =[512, 256, 128, 64, 32]


    # CSV 파일 초기화
    with open(output_file_fp8_baseline, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["input_token_size", "output_token_size", "batch_size", "latency"])


    
    # LLM 인스턴스 생성
    import gc  # 추가
    
    caputre_sizes = [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]


    for i, batch_size in enumerate(batch_sizes): 
        llm_custom = LLM(
            model=model_path,
            dtype="float16",
            max_num_batched_tokens = batch_size,
            max_num_seqs = batch_size,
            enable_prefix_caching=False,
            compilation_config = {
                "level": 3,
                "custom_ops": ["none"],
                "splitting_ops": [
                    "vllm.unified_attention",
                    "vllm.unified_attention_with_output"
                ],
                "full_cuda_graph": True,
                "use_inductor": True,
                "compile_sizes": caputre_sizes[i:],
                "use_cudagraph": True,
                "cudagraph_num_of_warmups": 2,
                "cudagraph_capture_sizes": caputre_sizes[i:],
                "max_capture_size": batch_size,
                "inductor_compile_config": {
                    "max_autotune": False
                },  
            },
            gpu_memory_utilization=0.97
        )

        with open(output_file_fp8_baseline, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Batch Size = ", batch_size, "", ""])
        
        for output_token_size in output_token_sizes:
            with open(output_file_fp8_baseline, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([f"Output Token Size = {output_token_size}", "", "", ""])

            sampling_params = SamplingParams(max_tokens=output_token_size, temperature=0, ignore_eos=True)

            for input_token_size in input_token_sizes:
                prompt_text = "A" * input_token_size
                original_prompts = [prompt_text]

                print(f"Running baseline inference: Batch={batch_size}, Input={input_token_size} tokens, Output={output_token_size} tokens...")
                prompts = original_prompts * batch_size
                run_inference_benchmark(
                    llm_custom,
                    output_file_fp8_baseline,
                    prompts,
                    sampling_params,
                    input_token_size,
                    output_token_size,
                    batch_size,
                    verbose=True
                )

        # 🔥 여기서 LLM 해제 + 캐시도 클리어
        del llm_custom
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
