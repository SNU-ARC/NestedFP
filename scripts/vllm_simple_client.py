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
import re
from dataclasses import dataclass, field
import aiohttp
from transformers import AutoTokenizer
import random
import matplotlib.pyplot as plt

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
    # Added iteration statistics
    iteration_data: list[dict] = field(default_factory=list)

# Global variables
ttft_graph = {'iteration_step': [], 'ttft': []}
iter_tpot_graph = {}  # {iteration_total: [token_latencies]}
iter_kv_graph = {}  # {iteration_total: [kv_cache_usage]}
iter_kv_gb_graph = {}  # {iteration_total: [kv_cache_usage_gb]}
iter_kv_total_capacity_graph = {}  # {iteration_total: [kv_cache_total_capacity]}
iter_num_prefill_graph = {}  # {iteration_total: [num_prefill]}
iter_num_decode_graph = {}  # {iteration_total: [num_decode]}

# Global variable for iteration details
iteration_details = {}  # {iteration_total: {...}}
iteration_lock = asyncio.Lock()  # async 환경에서 안전한 접근을 위한 락

async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
) -> RequestFuncOutput:
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
        previous_timestamp = None  # 🆕 ITL 계산을 위한 이전 timestamp 추적
        
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
                                timestamp = choices[0].get("iteration_timestamp")
                                iteration_total = choices[0].get("iteration_total")
                                kv_cache_usage = choices[0].get("kv_cache_usage")
                                kv_cache_usage_gb = choices[0].get("kv_cache_usage_gb")
                                kv_cache_total_capacity = choices[0].get("kv_cache_total_capacity")
                                num_prefill = choices[0].get("num_prefill")
                                num_decode = choices[0].get("num_decode")
                                # New scheduling information from server
                                total_scheduled_requests = choices[0].get("total_scheduled_requests")
                                total_scheduled_tokens = choices[0].get("total_scheduled_tokens")
                                prefill_requests = choices[0].get("prefill_requests") 
                                decode_requests = choices[0].get("decode_requests")
                                prefill_tokens = choices[0].get("prefill_tokens")
                                decode_tokens = choices[0].get("decode_tokens")
                                request_details = choices[0].get("request_details", [])
                                
                                # iteration 정보 저장 (기존 - RequestFuncOutput용)
                                if timestamp is not None and iteration_total is not None:
                                    output.iteration_data.append({
                                        "iteration_total": iteration_total,
                                        "timestamp": timestamp,
                                        "kv_cache_usage": kv_cache_usage,
                                        "text": text or ""
                                    })
                                
                                # 🆕 단순화된 iteration별 상세 정보 수집
                                if iteration_total is not None:
                                    async with iteration_lock:
                                        if iteration_total not in iteration_details:
                                            iteration_details[iteration_total] = {
                                                "iteration_total": iteration_total,
                                                "timestamp": timestamp,
                                                "tokens_generated": 0,
                                                "kv_cache_usage": kv_cache_usage,
                                                "kv_cache_usage_gb": kv_cache_usage_gb,
                                                "kv_cache_total_capacity": kv_cache_total_capacity,
                                                "total_scheduled_requests": total_scheduled_requests,
                                                "total_scheduled_tokens": total_scheduled_tokens,
                                                "prefill_requests": prefill_requests,
                                                "decode_requests": decode_requests,
                                                "prefill_tokens": prefill_tokens,
                                                "decode_tokens": decode_tokens,
                                                "request_details": request_details,
                                                "itl": None,  # 🆕 단일 ITL 값
                                            }
                                        
                                        # 기존 iteration이면 최신 정보로 업데이트
                                        iter_data = iteration_details[iteration_total]
                                        iter_data["timestamp"] = timestamp
                                        iter_data["tokens_generated"] += 1
                                        
                                        # 🆕 ITL 계산 (이전 토큰과의 시간차)
                                        if previous_timestamp is not None:
                                            itl = timestamp - previous_timestamp
                                            iter_data["itl"] = itl
                                
                                output.token_arrival_times.append(timestamp)
                                
                                if not first_chunk_received:
                                    first_chunk_received = True
                                    ttft = time.perf_counter() - st  # ✅ 클라이언트 기준
                                    output.ttft = ttft
                                    ttft_graph['iteration_step'].append(iteration_total)
                                    ttft_graph['ttft'].append(ttft)
                                else:
                                    output.itl.append(timestamp - previous_timestamp if previous_timestamp else 0)
                                    
                                # 🔄 기존 그래프 데이터 수집 (호환성 유지)
                                if iteration_total is not None and previous_timestamp is not None:
                                    token_latency = timestamp - previous_timestamp
                                    if iteration_total not in iter_tpot_graph:
                                        iter_tpot_graph[iteration_total] = []
                                    iter_tpot_graph[iteration_total].append(token_latency)
                                    
                                iter_kv_graph[iteration_total] = [kv_cache_usage]
                                iter_kv_gb_graph[iteration_total] = [kv_cache_usage_gb]
                                iter_kv_total_capacity_graph[iteration_total] = [kv_cache_total_capacity]
                                iter_num_prefill_graph[iteration_total] = [num_prefill]
                                iter_num_decode_graph[iteration_total] = [num_decode]

                                # 🆕 이전 timestamp 업데이트
                                if timestamp is not None:
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


# 🆕 iteration 상세 정보 후처리 함수
def process_iteration_details():
    """수집된 iteration 정보를 후처리하여 최종 형태로 변환"""
    processed_iterations = []
    
    for iteration_total in sorted(iteration_details.keys()):
        iter_data = iteration_details[iteration_total]
        
        # 🆕 단순화된 구조
        processed_iter = {
            "iteration_total": iteration_total,
            "timestamp": iter_data["timestamp"],  # 🆕 단일 timestamp
            "tokens_generated": iter_data["tokens_generated"],
            
            # 스케줄링 정보
            "total_scheduled_requests": iter_data["total_scheduled_requests"],
            "total_scheduled_tokens": iter_data["total_scheduled_tokens"],
            "prefill_requests": iter_data["prefill_requests"], 
            "decode_requests": iter_data["decode_requests"],
            "prefill_tokens": iter_data["prefill_tokens"],
            "decode_tokens": iter_data["decode_tokens"],
            
            # KV cache 정보
            "kv_cache_usage": iter_data["kv_cache_usage"],
            "kv_cache_usage_gb": iter_data["kv_cache_usage_gb"],
            "kv_cache_total_capacity": iter_data["kv_cache_total_capacity"],
            # 🆕 ITL (Inter-Token Latency) - 단일 값
            "itl": iter_data["itl"],
            # 요청별 세부 정보
            "request_details": iter_data["request_details"]
        }
        
        processed_iterations.append(processed_iter)
    
    return processed_iterations


# 기존 함수들은 그대로 유지...
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
    Trace 데이터를 로드하고 필터링
    
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

async def execute_single_request_with_prompt(request_input, prompt, api_url, model_name, request_id):
    """미리 생성된 프롬프트를 사용하는 버전"""
    input_obj = RequestFuncInput(
        prompt=prompt,  # 미리 생성된 프롬프트 사용
        api_url=api_url,
        prompt_len=request_input['CONTEXT_TOKENS'],
        output_len=request_input['GENERATED_TOKENS'],
        model=model_name
    )
    
    result = await async_request_openai_completions(input_obj)
    return result, request_id

async def execute_trace_based_requests(trace_df, prompts, api_url, model_name):
    results = []
    start_time = time.perf_counter()
    
    print(f"Starting trace-based test with {len(trace_df)} requests")
    
    for idx, row in trace_df.iterrows():
        target_time = row['relative_time']
        
        # 정확한 시간까지 대기
        while True:
            current_time = time.perf_counter() - start_time
            if current_time >= target_time:
                break
            await asyncio.sleep(0.001)  # 1ms 간격으로 체크
        
        # 요청 전송
        task = asyncio.create_task(
            execute_single_request_with_prompt(row, prompts[idx], api_url, model_name, idx)
        )
        results.append((task, row, idx))
        
        send_time = time.perf_counter() - start_time
        print(f"Request {idx+1}/{len(trace_df)} sent at {send_time:.6f}s (target: {target_time:.6f}s)")
    
    print(f"All requests sent. Waiting for responses...")
    
    final_results = []
    for task, row, request_id in results:
        result, _ = await task
        
        actual_generated_tokens = len(result.itl) + 1 if result.success and result.itl else 0
        
        final_results.append({
            'request_id': request_id,
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
            'itl': result.itl,
            'iteration_data': result.iteration_data
        })
    
    return final_results

async def run_experiment(trace_file, api_url, model_name, num_requests=None, duration_minutes=None, middle_ratio=0.8):
    """
    실험 실행
    
    Args:
        trace_file: trace 파일 경로
        api_url: API URL
        model_name: 모델 이름
        num_requests: 사용할 요청 개수 (legacy)
        duration_minutes: 실험 지속 시간 (분 단위)
        middle_ratio: 성능 분석용 중간 구간 비율
    """
    print(f"\n--- Running trace-based experiment ---")
    
    # 🆕 전역 변수 초기화
    global iteration_details
    iteration_details.clear()
    
    # trace 데이터 로드
    trace_df = load_trace_data(trace_file, num_requests=num_requests, duration_minutes=duration_minutes)
    
    # 실험 정보 출력
    if duration_minutes is not None:
        print(f"Experiment setup: {duration_minutes} minutes, {len(trace_df)} requests")
    elif num_requests is not None:
        print(f"Experiment setup: {num_requests} requests, {trace_df['relative_time'].max()/60:.2f} minutes")
    else:
        print(f"Experiment setup: {len(trace_df)} requests, {trace_df['relative_time'].max()/60:.2f} minutes")
    
    prompts = pregenerate_prompts(trace_df, tokenizer_name=model_name)
    
    experiment_start_time = time.perf_counter()
    print(f"Experiment started at {experiment_start_time:.6f}s")
    results = await execute_trace_based_requests(trace_df, prompts, api_url, model_name)
    experiment_end_time = time.perf_counter()
    print(f"Experiment completed at {experiment_end_time:.6f}s")
    
    # Request 별 결과 저장
    results_file = f"benchmark_request.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results with iteration data saved to {results_file}")
    
    
    # Iteration 별 결과 저장.
    processed_iterations = process_iteration_details()
    iteration_file = f"benchmark_iteration.json"
    with open(iteration_file, 'w') as f:
        json.dump(processed_iterations, f, indent=2)
    print(f"Iteration details saved to {iteration_file}")
    
    
    # 🆕 iteration 통계 출력
    print(f"\n--- Iteration Statistics ---")
    print(f"Total iterations executed: {len(processed_iterations)}")
    if processed_iterations:
        total_tokens = sum(iter_data['tokens_generated'] for iter_data in processed_iterations)
        avg_tokens_per_iter = total_tokens / len(processed_iterations)
        print(f"Total tokens generated: {total_tokens}")
        print(f"Average tokens per iteration: {avg_tokens_per_iter:.2f}")
        
        # 스케줄링 통계
        avg_scheduled_reqs = np.mean([iter_data.get('total_scheduled_requests', 0) for iter_data in processed_iterations if iter_data.get('total_scheduled_requests') is not None])
        avg_scheduled_tokens = np.mean([iter_data.get('total_scheduled_tokens', 0) for iter_data in processed_iterations if iter_data.get('total_scheduled_tokens') is not None])
        avg_prefill_reqs = np.mean([iter_data.get('prefill_requests', 0) for iter_data in processed_iterations if iter_data.get('prefill_requests') is not None])
        avg_prefill_tokens = np.mean([iter_data.get('prefill_tokens', 0) for iter_data in processed_iterations if iter_data.get('prefill_tokens') is not None])
        avg_decode_reqs = np.mean([iter_data.get('decode_requests', 0) for iter_data in processed_iterations if iter_data.get('decode_requests') is not None])
        avg_decode_tokens = np.mean([iter_data.get('decode_tokens', 0) for iter_data in processed_iterations if iter_data.get('decode_tokens') is not None])
        
        avg_kv_usage = np.mean([iter_data.get('kv_cache_usage', 0) for iter_data in processed_iterations if iter_data.get('kv_cache_usage') is not None])
        avg_kv_usage_gb = np.mean([iter_data.get('kv_cache_usage_gb', 0) for iter_data in processed_iterations if iter_data.get('kv_cache_usage_gb') is not None])
        avg_kv_total_capacity = np.mean([iter_data.get('kv_cache_total_capacity', 0) for iter_data in processed_iterations if iter_data.get('kv_cache_total_capacity') is not None])
        
        
        print(f"Average scheduled requests per iteration: {avg_scheduled_reqs:.2f}")
        print(f"Average scheduled tokens per iteration: {avg_scheduled_tokens:.2f}")
        print(f"Average prefill requests per iteration: {avg_prefill_reqs:.2f}")
        print(f"Average prefill tokens per iteration: {avg_prefill_tokens:.2f}")
        print(f"Average decode requests per iteration: {avg_decode_reqs:.2f}")
        print(f"Average decode tokens per iteration: {avg_decode_tokens:.2f}")
        
        print(f"Average kv cache usage per iteration: {avg_kv_usage:.4f}"
              f" (GB: {avg_kv_usage_gb:.4f}, Total Capacity: {avg_kv_total_capacity:.4f})")
    
    # iteration 데이터 요약 출력 (기존)
    total_tokens = sum(len(r['iteration_data']) for r in results if r['success'])
    print(f"Total tokens generated: {total_tokens}")
    print(f"Total Experiment Time: {experiment_end_time - experiment_start_time:.2f}s")
    
    # Store the Total Experiment Time and Total Requests
    with open("benchmark_summary.json", 'w') as f:
        summary = {
            "total_experiment_time": experiment_end_time - experiment_start_time,
            "total_requests": len(results),
            "successful_requests": sum(1 for r in results if r['success']),
            "failed_requests": sum(1 for r in results if not r['success'])
        }
        json.dump(summary, f, indent=2)
    print(f"Summary saved to benchmark_summary.json")
    
    return results

async def run_throughput_test(
    api_url,
    model_name,
    input_len=512,
    output_len=32,
    num_requests=512,
    nestedfp=False,
):
    print(f"\n--- Running Throughput Test ---")
    print(f"Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Nested FP: {nestedfp}")
    print(f"  Input length: {input_len} tokens")
    print(f"  Output length: {output_len} tokens")
    print(f"  Number of requests: {num_requests}")

    global iteration_details
    iteration_details.clear()

    print(f"\nGenerating prompts with tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    words = [
        "hello",
        "world",
        "test",
        "data",
        "model",
        "training",
        "computer",
        "science",
        "artificial",
        "intelligence",
        "machine",
        "learning",
        "neural",
        "network",
        "transformer",
        "attention",
        "embedding",
        "layer",
        "parameter",
        "gradient",
        "system",
        "processing",
        "algorithm",
        "function",
        "method",
        "class",
        "object",
        "memory",
        "storage",
        "database",
        "server",
        "client",
        "protocol",
        "interface",
    ]

    prompts = []
    for i in range(num_requests):
        random.seed(i)
        selected_words = random.choices(words, k=input_len*2)
        prompt = " ".join(selected_words)
        encoded = tokenizer.encode(prompt, add_special_tokens=False)
        if len(encoded) < input_len:
            while len(encoded) < input_len:
                additional_word = random.choice(words)
                test_prompt = prompt + " " + additional_word
                test_encoded = tokenizer.encode(test_prompt, add_special_tokens=False)
                if len(test_encoded) <= input_len:
                    prompt = test_prompt
                    encoded = test_encoded
                else:
                    break
        truncated_tokens = encoded[:input_len]
        assert len(truncated_tokens) == input_len
        prompt = tokenizer.decode(truncated_tokens)
        prompts.append(prompt)

    print(f"All prompts generated ({len(prompts)})")

    print(f"\nSending all {num_requests} requests simultaneously...")
    experiment_start_time = time.perf_counter()

    tasks = []
    for i in range(num_requests):
        input_obj = RequestFuncInput(
            prompt=prompts[i],
            api_url=api_url,
            prompt_len=input_len,
            output_len=output_len,
            model=model_name,
        )
        task = asyncio.create_task(async_request_openai_completions(input_obj))
        tasks.append((task, i))

    print(f"All requests sent at {experiment_start_time:.6f}s")
    print("Waiting for all responses...")

    results = []
    for task, request_id in tasks:
        result = await task
        actual_generated_tokens = (
            len(result.itl) + 1 if result.success and result.itl else 0
        )
        results.append(
            {
                "request_id": request_id,
                "context_tokens": input_len,
                "generated_tokens": output_len,
                "actual_generated_tokens": actual_generated_tokens,
                "success": result.success,
                "latency": result.latency,
                "ttft": result.ttft,
                "tpot": result.tpot,
                "error": result.error,
                "request_sent_time": result.request_sent_time,
                "request_completed_time": result.request_completed_time,
                "token_arrival_times": result.token_arrival_times,
                "itl": result.itl,
                "iteration_data": result.iteration_data,
            }
        )

    experiment_end_time = time.perf_counter()
    total_experiment_time = experiment_end_time - experiment_start_time

    print(f"\nAll requests completed at {experiment_end_time:.6f}s")
    print(f"Total experiment time: {total_experiment_time:.2f}s")

    # Statistics
    success_count = sum(1 for r in results if r["success"])
    latencies = [r["latency"] for r in results if r["success"]]
    ttfts = [r["ttft"] for r in results if r["success"] and r["ttft"] > 0]
    tpots = [r["tpot"] for r in results if r["success"] and r["tpot"] > 0]

    total_tokens = sum(r["actual_generated_tokens"] for r in results if r["success"])
    throughput = total_tokens / total_experiment_time if total_experiment_time > 0 else 0

    assert num_requests == success_count

    summary = {
        "model": model_name,
        "nestedfp": nestedfp,
        "input_length": input_len,
        "output_length": output_len,
        "num_requests": num_requests,
        "successful_requests": success_count,
        "failed_requests": num_requests - success_count,
        "avg_e2e_latency": float(np.mean(latencies)) if latencies else 0.0,
        "p90_e2e_latency": float(np.percentile(latencies, 90)) if latencies else 0.0,
        "p99_e2e_latency": float(np.percentile(latencies, 99)) if latencies else 0.0,
        "avg_ttft": float(np.mean(ttfts)) if ttfts else 0.0,
        "avg_tpot": float(np.mean(tpots)) if tpots else 0.0,
        "total_tokens": total_tokens,
        "throughput_tokens_per_sec": throughput,
        "total_experiment_time": total_experiment_time,
    }

    return summary

async def run_throughput_sweep(
    api_url,
    model_name,
    input_output_combinations=None,
    batch_sizes=[32, 64, 128, 256, 512],
    nestedfp=False,
):    
    print(f"\n{'='*80}")
    print("THROUGHPUT SWEEP TEST - MULTIPLE CONFIGURATIONS")
    print(f"{'='*80}")
    print(f"Model: {model_name}")
    print(f"Nested FP: {nestedfp}")
    print(f"Input/Output combinations: {input_output_combinations}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"{'='*80}\n")

    all_results = []
    
    # Loop through each (input, output) combination
    for input_len, output_len in input_output_combinations:
        print(f"\n{'#'*80}")
        print(f"# Testing configuration: Input={input_len}, Output={output_len}")
        print(f"{'#'*80}")
        
        # Loop through each batch size for this combination
        for batch_size in batch_sizes:
            print(f"\n>>> Running throughput test for batch size = {batch_size}")
            summary = await run_throughput_test(
                api_url=api_url,
                model_name=model_name,
                input_len=input_len,
                output_len=output_len,
                num_requests=batch_size,
                nestedfp=nestedfp,
            )
            
            summary["input_len"] = input_len
            summary["output_len"] = output_len
            summary["batch_size"] = batch_size
            
            all_results.append(summary)

    model_tag = os.path.basename(model_name.rstrip("/"))
    sweep_file = f"throughput_sweep_{model_tag}_{nestedfp}.json"

    output_data = {
        "model": model_name,
        "model_short_name": model_tag,
        "nestedfp": nestedfp,
        "test_configurations": {
            "input_output_combinations": input_output_combinations,
            "batch_sizes": batch_sizes
        },
        "results": all_results
    }
    
    with open(sweep_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n{'='*80}")
    print(f"✅ All sweep results saved to {sweep_file}")
    print(f"Total configurations tested: {len(all_results)}")
    print(f"{'='*80}\n")

    # Print summary table
    print("\n=== Throughput Sweep Summary ===")
    print(f"{'Input':>6} | {'Output':>6} | {'Batch':>6} | {'Throughput':>12} | {'Avg E2E':>10} | {'Total Tokens':>12}")
    print(f"{'(tok)':>6} | {'(tok)':>6} | {'Size':>6} | {'(tok/s)':>12} | {'Latency(s)':>10} | {'':>12}")
    print("-" * 85)
    for entry in all_results:
        print(
            f"{entry['input_length']:>6} | {entry['output_length']:>6} | {entry['num_requests']:>6} | "
            f"{entry['throughput_tokens_per_sec']:>12.2f} | {entry['avg_e2e_latency']:>10.3f} | "
            f"{entry['total_tokens']:>12}"
        )
    print("-" * 85)
    return all_results

async def main():
    parser = argparse.ArgumentParser(description="vLLM client")
    
    # Test mode selection
    parser.add_argument("--test-mode",
                       choices=["trace", "throughput"],
                       default="trace",
                       help="Test mode: 'trace' for trace-based test, 'throughput' for throughput test")
    
    # Trace mode arguments
    parser.add_argument("--trace-file",
                       default="../trace/azure_conv_0514_1400_20min_10.0x_tc.csv",
                       help="Path to trace CSV file (for trace mode)")
    parser.add_argument("--num-requests",
                       type=int,
                       default=None,
                       help="Number of requests to use (legacy mode, for trace mode)")
    parser.add_argument("--duration-minutes",
                       type=float,
                       default=None,
                       help="Duration of experiment in minutes (for trace mode)")
    parser.add_argument("--middle-ratio",
                       type=float,
                       default=0.7,
                       help="Ratio of middle data to use for performance stats (for trace mode)")
    
    # Common arguments
    parser.add_argument("--api-url",
                       default="http://0.0.0.0:8000/v1/completions",
                       help="vLLM server API URL")
    parser.add_argument("--model",
                       default="/home/ubuntu/models/Llama-3.1-8B",
                       help="Model name or path")
    
    # Throughput test arguments
    parser.add_argument("--throughput-input-output-combinations",
                       type=str,
                       default=None,
                       help="Comma-separated list of input,output pairs (e.g., '128,32;256,32;512,32')")
    parser.add_argument("--throughput-batch-sizes",
                       type=str,
                       default="32,64,128,256,512",
                       help="Comma-separated list of batch sizes (default: '32,64,128,256,512')")
    
    parser.add_argument("--nestedfp", 
                       action="store_true",
                       help="Use nested FP16 model if set")
    
    args = parser.parse_args()
    
    if not args.api_url.endswith(("completions", "profile")):
        if not args.api_url.endswith('/'):
            args.api_url += '/v1/completions'
        else:
            args.api_url += 'v1/completions'
    
    try:
        if args.test_mode == "throughput":
            # Parse input/output combinations
            if args.throughput_input_output_combinations:
                combinations = []
                for pair in args.throughput_input_output_combinations.split(';'):
                    inp, out = map(int, pair.split(','))
                    combinations.append((inp, out))
            else:
                combinations = [
                    (32, 32), (32, 512), (1024, 32), (1024, 512)
                ]
            
            # Parse batch sizes
            batch_sizes = [int(x) for x in args.throughput_batch_sizes.split(',')]
            
            # Run throughput test 
            await run_throughput_sweep(
                api_url=args.api_url,
                model_name=args.model,
                input_output_combinations=combinations,
                batch_sizes=batch_sizes,
                nestedfp=args.nestedfp,
            )
        else:
            # Trace-based Test 실행
            # 두 파라미터가 모두 없는 경우 기본값 설정
            if args.num_requests is None and args.duration_minutes is None:
                args.duration_minutes = 20.0  # 기본값: 20 minutes
                print("No duration or num_requests specified, using default: 20 minutes")
            
            # 두 파라미터가 모두 제공된 경우 에러
            if args.num_requests is not None and args.duration_minutes is not None:
                print("Error: Cannot specify both --num-requests and --duration-minutes. Please use only one.")
                return
            
            await run_experiment(
                args.trace_file, 
                args.api_url, 
                args.model, 
                num_requests=args.num_requests,
                duration_minutes=args.duration_minutes,
                middle_ratio=args.middle_ratio
            )
    
    except ValueError as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    asyncio.run(main())
