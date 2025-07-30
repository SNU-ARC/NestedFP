import os
import subprocess
import torch
import sys
import time
import csv
from vllm import LLM, SamplingParams

# import sys
# sys.path.insert(0, "/disk/revision/vllm")  # vllm 소스 디렉토리의 상위 경로

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

def test_model_generation(model_path, output_file):
    """
    Test the model's generation capabilities and save results to CSV.
    """
    # Define meaningful prompts
    prompts = [
        "Explain the future of artificial intelligence.",
        "What are the most famous traditional foods in Korean cuisine?",
        "Why is Python popular among programming languages?",
        "Explain the impact of climate change on ecosystems.",
        "What are the basic principles of quantum computing?"
    ]
    
    # List to store results
    results = []
    
    # Initialize CSV file
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["prompt", "response", "input_tokens", "output_tokens", "generation_time"])
    
    # Load the model (with safe settings)
    try:
        print(f"Loading model: {model_path}")
        llm = LLM(
            model=model_path,
            dtype="float16",
            quantization="dualfp",
            # max_model_len=4096,       # Limit context length
            tensor_parallel_size=4,
            enforce_eager=True        # Eager execution for better debugging
        )
        
        # Set sampling parameters
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=256          # Safer output length
        )
        
        # Run generation for each prompt
        for prompt in prompts:
            print(f"Processing prompt: {prompt[:30]}...")
            
            try:
                # Measure generation time
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                
                outputs = llm.generate([prompt], sampling_params)
                
                torch.cuda.synchronize()
                end_time = time.perf_counter()
                generation_time = end_time - start_time
                
                # Parse results
                if outputs and len(outputs) > 0:
                    generated_text = outputs[0].outputs[0].text
                    input_token_count = len(outputs[0].prompt_token_ids)
                    output_token_count = len(outputs[0].outputs[0].token_ids)
                    
                    # Store results
                    results.append({
                        "prompt": prompt,
                        "response": generated_text,
                        "input_tokens": input_token_count,
                        "output_tokens": output_token_count,
                        "generation_time": generation_time
                    })
                    
                    # Log results to CSV
                    with open(output_file, mode="a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow([
                            prompt, 
                            generated_text, 
                            input_token_count, 
                            output_token_count, 
                            generation_time
                        ])
                    
                    print(f"Generation complete: Input tokens {input_token_count}, Output tokens {output_token_count}, Time: {generation_time:.4f}s")
                    print(f"Response: {generated_text[:100]}...\n")
                
            except Exception as e:
                print(f"Error during generation for prompt '{prompt[:30]}...': {str(e)}")
                # Continue despite errors
        
        return results
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return []

def main():
    # Select the GPU with the most available memory
    idle_gpu = get_idle_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(idle_gpu)
    print(f"Using GPU {idle_gpu}")
    
    # Set model path and output file
    # model_path = "/disk2/models/Mistral-Small-24B-Instruct-2501"
    # model_path = "/disk2/models/DeepSeek-R1-Distill-Llama-70B""
    model_path = "/disk2/models/Llama-3.1-70B"
    output_file = "test_results.csv"

    # Safely handle exceptions
    try:
        # Clear GPU cache to minimize memory usage
        torch.cuda.empty_cache()
        
        # Run model testing
        results = test_model_generation(model_path, output_file)
        
        # Print summary info
        if results:
            avg_time = sum(r["generation_time"] for r in results) / len(results)
            avg_output_tokens = sum(r["output_tokens"] for r in results) / len(results)
            
            print("\nTest Summary:")
            print(f"Number of prompts tested: {len(results)}")
            print(f"Average generation time: {avg_time:.4f}s")
            print(f"Average output tokens: {avg_output_tokens:.1f}")
            print(f"Results saved to: {output_file}")
        else:
            print("No test results available.")
            
    except Exception as e:
        print(f"Exception during execution: {str(e)}")

if __name__ == "__main__":
    # Improve stability in multiprocessing environments
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    
    # Run main function
    main()