import asyncio
import argparse
import pandas as pd
import numpy as np
import time
from typing import Optional
from datetime import datetime
import json
import os
import sys
import traceback
from dataclasses import dataclass, field
import aiohttp
from transformers import AutoTokenizer
import random

@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    model_name: Optional[str] = None

@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0
    itl: list[float] = field(default_factory=list)
    tpot: float = 0.0
    prompt_len: int = 0
    error: str = ""
    token_arrival_times: list[float] = field(default_factory=list)
    request_sent_time: float = 0.0
    request_completed_time: float = 0.0

async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
) -> RequestFuncOutput:
    """단순화된 OpenAI Completions API 요청 함수 (SLO 측정용)"""
    api_url = request_func_input.api_url
    assert api_url.endswith(("completions", "profile")), \
        "OpenAI Completions API URL must end with 'completions' or 'profile'."

    AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
    async with aiohttp.ClientSession(trust_env=True, timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "model": request_func_input.model_name or request_func_input.model,
            "prompt": request_func_input.prompt,
            "temperature": 0.0,
            "repetition_penalty": 1.0,
            "max_tokens": request_func_input.output_len,
            "stream": True,
            "stream_options": {"include_usage": True},
            "ignore_eos": True,
        }
        
        headers = {}
        openai_api_key = os.environ.get('OPENAI_API_KEY')
        if openai_api_key:
            headers["Authorization"] = f"Bearer {openai_api_key}"

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        st = time.perf_counter()
        output.request_sent_time = st
        previous_timestamp = None
        
        try:
            async with session.post(url=api_url, json=payload, headers=headers) as response:
                if response.status == 200:
                    first_chunk_received = False
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = chunk_bytes.decode("utf-8").removeprefix("data: ")
                        if chunk != "[DONE]":
                            data = json.loads(chunk)

                            if choices := data.get("choices"):
                                text = choices[0].get("text")
                                timestamp = choices[0].get("iteration_timestamp", time.perf_counter())
                                
                                output.token_arrival_times.append(timestamp)
                                
                                if not first_chunk_received:
                                    first_chunk_received = True
                                    ttft = timestamp - st
                                    output.ttft = ttft
                                else:
                                    if previous_timestamp is not None:
                                        output.itl.append(timestamp - previous_timestamp)
                                    
                                previous_timestamp = timestamp
                                generated_text += text or ""
                    
                    output.request_completed_time = time.perf_counter()
                    
                    if first_chunk_received:
                        output.success = True
                        output.generated_text = generated_text
                        output.latency = output.request_completed_time - st
                        if output.itl:
                            output.tpot = sum(output.itl) / len(output.itl)
                    else:
                        output.success = False
                        output.error = "Never received a valid chunk to calculate TTFT."
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

        return output

def pregenerate_prompts(trace_df, tokenizer_name="Qwen/Qwen2.5-7B"):
    """효율적으로 정확한 토큰 길이의 프롬프트 생성"""
    print(f"Pre-generating prompts using tokenizer: {tokenizer_name}")
    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
   
    # 간단한 단어 풀
    words = ["hello", "world", "test", "data", "model", "training", "computer", "science",
            "artificial", "intelligence", "machine", "learning", "neural", "network",
            "transformer", "attention", "embedding", "layer", "parameter", "gradient",
            "system", "processing", "algorithm", "function", "method", "class", "object",
            "memory", "storage", "database", "server", "client", "protocol", "interface"]
   
    prompts = []
    
    for idx, row in trace_df.iterrows():
        target_tokens = row['CONTEXT_TOKENS']
        
        # request별 다른 시드 사용 (KV Cache 다양성 확보)
        random.seed(idx)
        
        # 충분히 긴 프롬프트 생성 (target_tokens의 1.5배 정도)
        selected_words = random.choices(words, k=target_tokens * 2)
        prompt = " ".join(selected_words)
        
        # 토큰 길이 확인 및 조정
        encoded = tokenizer.encode(prompt, add_special_tokens=False)
        
        if len(encoded) > target_tokens:
            # 토큰 단위로 정확히 자르기
            truncated_tokens = encoded[:target_tokens]
            prompt = tokenizer.decode(truncated_tokens)
        elif len(encoded) < target_tokens:
            # 부족한 경우 단어 더 추가
            while len(encoded) < target_tokens:
                additional_word = random.choice(words)
                test_prompt = prompt + " " + additional_word
                test_encoded = tokenizer.encode(test_prompt, add_special_tokens=False)
                if len(test_encoded) <= target_tokens:
                    prompt = test_prompt
                    encoded = test_encoded
                else:
                    break
        
        prompts.append(prompt)
        
        if (idx + 1) % 100 == 0:
            print(f"Generated {idx + 1}/{len(trace_df)} prompts")
    
    # 검증
    print("\nValidating generated prompts...")
    mismatches = 0
    for i, (prompt, target_length) in enumerate(zip(prompts, trace_df['CONTEXT_TOKENS'])):
        actual_length = len(tokenizer.encode(prompt, add_special_tokens=False))
        if actual_length != target_length:
            mismatches += 1
            if mismatches <= 5:
                print(f"Mismatch at index {i}: target={target_length}, actual={actual_length}")
    
    generation_time = time.time() - start_time
    print(f"Prompt generation completed in {generation_time:.2f}s")
    print(f"Token length mismatches: {mismatches}/{len(prompts)}")
    
    return prompts

def load_trace_data(file_path, num_requests=None, duration_minutes=None):
    """
    Trace 데이터를 로드하고 필터링 (SLO 측정용)
    
    Args:
        file_path: CSV 파일 경로
        num_requests: 사용할 요청 개수 (legacy, duration_minutes와 함께 사용 불가)
        duration_minutes: 실험 지속 시간 (분 단위)
    """
    df = pd.read_csv(file_path)
    df['TIMESTAMP'] = pd.to_datetime(df.iloc[:, 0])
    df.columns = ['TIMESTAMP', 'CONTEXT_TOKENS', 'GENERATED_TOKENS']
    
    # 상대 시간 계산
    first_timestamp = df['TIMESTAMP'].min()
    df['relative_time'] = (df['TIMESTAMP'] - first_timestamp).dt.total_seconds()
    
    # trace 데이터의 총 지속 시간 계산
    total_duration_seconds = df['relative_time'].max()
    total_duration_minutes = total_duration_seconds / 60
    
    print(f"Trace file loaded: {len(df)} total requests")
    print(f"Trace duration: {total_duration_minutes:.2f} minutes ({total_duration_seconds:.2f} seconds)")
    
    # 두 파라미터가 모두 제공된 경우 에러
    if num_requests is not None and duration_minutes is not None:
        raise ValueError("Cannot specify both num_requests and duration_minutes. Please use only one.")
    
    # duration_minutes 기준으로 필터링
    if duration_minutes is not None:
        target_duration_seconds = duration_minutes * 60
        
        # 요청한 시간이 trace 데이터보다 긴 경우 에러
        if target_duration_seconds > total_duration_seconds:
            raise ValueError(f"Requested duration ({duration_minutes} minutes) exceeds trace data duration ({total_duration_minutes:.2f} minutes)")
        
        # 지정된 시간 내의 요청만 필터링
        filtered_df = df[df['relative_time'] <= target_duration_seconds].copy()
        
        print(f"Using {duration_minutes} minutes of trace data: {len(filtered_df)} requests")
        print(f"Actual duration used: {filtered_df['relative_time'].max():.2f} seconds")
        
        return filtered_df
    
    # num_requests 기준으로 필터링 (legacy 지원)
    elif num_requests is not None:
        if len(df) > num_requests:
            df = df.head(num_requests)
            print(f"Using first {num_requests} requests from trace file")
            print(f"Duration of selected requests: {df['relative_time'].max():.2f} seconds ({df['relative_time'].max()/60:.2f} minutes)")
        else:
            print(f"Using all {len(df)} requests from trace file")
        
        return df
    
    # 아무것도 지정되지 않은 경우 모든 데이터 사용
    else:
        print(f"Using all {len(df)} requests from trace file")
        return df

async def execute_sequential_requests(trace_df, prompts, api_url, model_name):
    """순차적으로 요청을 실행하여 SLO 데이터 수집"""
    results = []
    
    print(f"Starting sequential SLO measurement with {len(trace_df)} requests")
    print("Note: Requests will be sent one by one, ignoring trace timestamps")
    
    start_time = time.perf_counter()
    
    for idx, row in trace_df.iterrows():
        print(f"Processing request {idx+1}/{len(trace_df)}")
        
        # 요청 생성
        input_obj = RequestFuncInput(
            prompt=prompts[idx],
            api_url=api_url,
            prompt_len=row['CONTEXT_TOKENS'],
            output_len=row['GENERATED_TOKENS'],
            model=model_name
        )
        
        # 요청 실행
        result = await async_request_openai_completions(input_obj)
        
        # 실제 생성된 토큰 수 계산
        actual_generated_tokens = len(result.itl) + 1 if result.success and result.itl else 0
        
        # 결과 저장
        request_result = {
            'request_id': idx,
            'context_tokens': row['CONTEXT_TOKENS'],
            'generated_tokens': row['GENERATED_TOKENS'],
            'actual_generated_tokens': actual_generated_tokens,
            'success': result.success,
            'latency': result.latency,
            'ttft': result.ttft,
            'tpot': result.tpot,
            'error': result.error,
            'request_sent_time': result.request_sent_time,
            'request_completed_time': result.request_completed_time,
            'token_arrival_times': result.token_arrival_times,
            'itl': result.itl
        }
        
        results.append(request_result)
        
        # 간단한 진행 상황 출력
        if result.success:
            print(f"  ✓ Success - TTFT: {result.ttft:.4f}s, TPOT: {result.tpot:.4f}s, Context Tokens: {row['CONTEXT_TOKENS']}, Generated Tokens: {actual_generated_tokens}")
        else:
            print(f"  ✗ Failed - Error: {result.error[:100]}...")
    
    end_time = time.perf_counter()
    print(f"\nAll requests completed in {end_time - start_time:.2f}s")
    
    return results

def calculate_slo_statistics(results):
    """SLO 계산을 위한 통계 정보 계산"""
    successful_results = [r for r in results if r['success']]
    total_requests = len(results)
    successful_requests = len(successful_results)
    
    print(f"\n--- SLO Statistics ---")
    print(f"Total requests: {total_requests}")
    print(f"Successful requests: {successful_requests}")
    print(f"Success rate: {successful_requests/total_requests*100:.2f}%")
    
    if not successful_results:
        print("No successful requests to calculate SLO statistics")
        return {}
    
    # TTFT 통계
    ttft_values = [r['ttft'] for r in successful_results if r['ttft'] > 0]
    tpot_values = [r['tpot'] for r in successful_results if r['tpot'] > 0]
    latency_values = [r['latency'] for r in successful_results]
    
    stats = {
        'total_requests': total_requests,
        'successful_requests': successful_requests,
        'success_rate': successful_requests/total_requests,
    }
    
    if ttft_values:
        stats['ttft'] = {
            'mean': np.mean(ttft_values),
            'median': np.median(ttft_values),
            'std': np.std(ttft_values),
            'min': np.min(ttft_values),
            'max': np.max(ttft_values),
            'p90': np.percentile(ttft_values, 90),
            'p95': np.percentile(ttft_values, 95),
            'p99': np.percentile(ttft_values, 99),
            'count': len(ttft_values)
        }
        
        print(f"\nTTFT Statistics:")
        print(f"  Mean: {stats['ttft']['mean']:.4f}s")
        print(f"  Median: {stats['ttft']['median']:.4f}s")
        print(f"  Std: {stats['ttft']['std']:.4f}s")
        print(f"  Min: {stats['ttft']['min']:.4f}s")
        print(f"  Max: {stats['ttft']['max']:.4f}s")
        print(f"  P90: {stats['ttft']['p90']:.4f}s")
        print(f"  P95: {stats['ttft']['p95']:.4f}s")
        print(f"  P99: {stats['ttft']['p99']:.4f}s")
    
    if tpot_values:
        stats['tpot'] = {
            'mean': np.mean(tpot_values),
            'median': np.median(tpot_values),
            'std': np.std(tpot_values),
            'min': np.min(tpot_values),
            'max': np.max(tpot_values),
            'p90': np.percentile(tpot_values, 90),
            'p95': np.percentile(tpot_values, 95),
            'p99': np.percentile(tpot_values, 99),
            'count': len(tpot_values)
        }
        
        print(f"\nTPOT Statistics:")
        print(f"  Mean: {stats['tpot']['mean']:.4f}s")
        print(f"  Median: {stats['tpot']['median']:.4f}s")
        print(f"  Std: {stats['tpot']['std']:.4f}s")
        print(f"  Min: {stats['tpot']['min']:.4f}s")
        print(f"  Max: {stats['tpot']['max']:.4f}s")
        print(f"  P90: {stats['tpot']['p90']:.4f}s")
        print(f"  P95: {stats['tpot']['p95']:.4f}s")
        print(f"  P99: {stats['tpot']['p99']:.4f}s")
    
    if latency_values:
        stats['latency'] = {
            'mean': np.mean(latency_values),
            'median': np.median(latency_values),
            'std': np.std(latency_values),
            'min': np.min(latency_values),
            'max': np.max(latency_values),
            'p90': np.percentile(latency_values, 90),
            'p95': np.percentile(latency_values, 95),
            'p99': np.percentile(latency_values, 99),
            'count': len(latency_values)
        }
        
        print(f"\nLatency Statistics:")
        print(f"  Mean: {stats['latency']['mean']:.4f}s")
        print(f"  Median: {stats['latency']['median']:.4f}s")
    
    # SLO 제안값 계산 (mean 기준)
    if 'ttft' in stats and 'tpot' in stats:
        slo_multipliers = [5, 10, 15]  # tight, medium, loose
        
        print(f"\n--- Suggested SLO Values ---")
        for multiplier in slo_multipliers:
            ttft_slo = stats['ttft']['mean'] * multiplier
            tpot_slo = stats['tpot']['mean'] * multiplier
            print(f"Multiplier {multiplier}x:")
            print(f"  TTFT SLO: {ttft_slo:.4f}s")
            print(f"  TPOT SLO: {tpot_slo:.4f}s")
    
    return stats

async def run_slo_experiment(trace_file, api_url, model_name, num_requests=None, duration_minutes=None):
    """SLO 측정 실험 실행"""
    print(f"\n--- Running SLO Measurement Experiment ---")
    print("This experiment sends requests sequentially to measure baseline TTFT/TPOT")
    
    # trace 데이터 로드
    trace_df = load_trace_data(trace_file, num_requests=num_requests, duration_minutes=duration_minutes)
    
    # 실험 정보 출력
    if duration_minutes is not None:
        print(f"SLO measurement setup: {duration_minutes} minutes worth of trace data, {len(trace_df)} requests")
    elif num_requests is not None:
        print(f"SLO measurement setup: {num_requests} requests")
    else:
        print(f"SLO measurement setup: {len(trace_df)} requests")
    
    # 프롬프트 생성
    prompts = pregenerate_prompts(trace_df, tokenizer_name=model_name)
    
    # 순차적 요청 실행
    experiment_start_time = time.perf_counter()
    results = await execute_sequential_requests(trace_df, prompts, api_url, model_name)
    experiment_end_time = time.perf_counter()
    
    print(f"Total experiment time: {experiment_end_time - experiment_start_time:.2f}s")
    
    # 통계 계산
    stats = calculate_slo_statistics(results)
    
    # 결과 저장
    output_data = {
        'experiment_info': {
            'trace_file': trace_file,
            'api_url': api_url,
            'model_name': model_name,
            'num_requests_used': len(trace_df),
            'duration_minutes': duration_minutes,
            'experiment_start_time': experiment_start_time,
            'experiment_end_time': experiment_end_time,
            'total_experiment_time': experiment_end_time - experiment_start_time
        },
        'statistics': stats,
        'results': results
    }
    
    # JSON 파일로 저장
    results_file = "benchmark_slo.json"
    with open(results_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSLO measurement results saved to {results_file}")
    
    return results, stats

async def main():
    parser = argparse.ArgumentParser(description="vLLM SLO measurement tool")
    parser.add_argument("--trace-file", 
                       default="/home2/omin/FAServe/online/trace/azure_conv_0514_1400_20min_15.0x_tc.csv",
                       help="Path to trace CSV file")
    parser.add_argument("--api-url", 
                       default="http://0.0.0.0:9000/v1/completions", 
                       help="vLLM server API URL")
    parser.add_argument("--model", 
                       default="Qwen/Qwen2.5-7B",
                       help="Model name or path")
    parser.add_argument("--num-requests", 
                       type=int, 
                       default=None,
                       help="Number of requests to use (legacy mode)")
    parser.add_argument("--duration-minutes", 
                       type=float, 
                       default=None,
                       help="Duration of trace data to use in minutes (preferred over num-requests)")
    
    args = parser.parse_args()
    
    # 두 파라미터가 모두 없는 경우 기본값 설정
    if args.num_requests is None and args.duration_minutes is None:
        # args.num_requests = 1000  # 기본값: 1000 requests for SLO measurement
        # print("No duration or num_requests specified, using default: 1000 requests")
        args.duration_minutes = 10  # 기본값: 10 minutes for SLO measurement
        print("No duration or num_requests specified, using default: 10 minutes")

    # 두 파라미터가 모두 제공된 경우 에러
    if args.num_requests is not None and args.duration_minutes is not None:
        print("Error: Cannot specify both --num-requests and --duration-minutes. Please use only one.")
        return
    
    # API URL 정규화
    if not args.api_url.endswith(("completions", "profile")):
        if not args.api_url.endswith('/'):
            args.api_url += '/v1/completions'
        else:
            args.api_url += 'v1/completions'
    
    try:
        await run_slo_experiment(
            args.trace_file, 
            args.api_url, 
            args.model, 
            num_requests=args.num_requests,
            duration_minutes=args.duration_minutes
        )
    except ValueError as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    asyncio.run(main())