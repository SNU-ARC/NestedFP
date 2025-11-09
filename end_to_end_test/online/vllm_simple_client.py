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
    # 새로 추가: iteration 정보
    iteration_data: list[dict] = field(default_factory=list)

# 전역 변수들
ttft_graph = {'iteration_step': [], 'ttft': []}
iter_tpot_graph = {}  # {iteration_total: [token_latencies]}
iter_kv_graph = {}  # {iteration_total: [kv_cache_usage]}
iter_kv_gb_graph = {}  # {iteration_total: [kv_cache_usage_gb]}
iter_kv_total_capacity_graph = {}  # {iteration_total: [kv_cache_total_capacity]}
iter_num_prefill_graph = {}  # {iteration_total: [num_prefill]}
iter_num_decode_graph = {}  # {iteration_total: [num_decode]}

# 🆕 iteration별 상세 정보 수집용 전역 변수
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
                                # 서버에서 온 새로운 스케줄링 정보들
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

# 기존 그래프 생성 함수들은 그대로 유지...
def save_ttft_plot(dic, x, y, file_name):
    df = pd.DataFrame(dic)
    if len(df) > 0:
        df = df.sort_values(x)
        plt.figure(figsize=(10, 6))
        plt.plot(df[x], df[y], '-', linewidth=1, alpha=0.6, color='lightblue')
        plt.scatter(df[x], df[y], s=50, alpha=0.9, color='red', edgecolors='darkred', linewidth=1)
        plt.xlabel('First Token Iteration Step')  # 의미있는 라벨
        plt.ylabel('TTFT [s]')
        plt.title('TTFT by First Token Iteration Step')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{file_name}.png', dpi=1000)
        plt.close()

def save_num_data_plots(iter_dict, file_name, data_name):
    """서버 iteration step별 num_prefill or num_decode 그래프 저장"""
    if not iter_dict:
        print(f"No {data_name} data to plot")
        return
    
    iterations = []
    latencies = []
    
    for iteration_step in sorted(iter_dict.keys()):
        step_latencies = iter_dict[iteration_step]
        for latency in step_latencies:
            iterations.append(iteration_step)
            latencies.append(latency)
    
    if not iterations:
        print(f"No valid {data_name} data to plot")
        return
    
    # 단일 그래프 생성 (선으로 연결)
    plt.figure(figsize=(12, 6))
    plt.plot(iterations, latencies, 'o-', markersize=4, linewidth=1, alpha=0.7, color='blue')
    plt.xlabel('Server Iteration Step')
    plt.ylabel(f'{data_name} [number]')
    plt.title(f'{data_name} by Iteration Step')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{file_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"{data_name} saved to {file_name}.png")
    
    # 간단한 통계 정보 출력
    print(f"\n{data_name} usage Statistics:")
    print(f"Total iteration steps executed: {len(iter_dict)}")  # 실제 실행된 iteration step 횟수
    print(f"Unique iteration step range: {min(iterations)} - {max(iterations)}")
    if latencies:
        print(f"Average {data_name} usage: {np.mean(latencies):.4f}s")
        print(f"Min {data_name} usage: {min(latencies):.4f}s")
        print(f"Max {data_name} usage: {max(latencies):.4f}s")

def save_kv_plots(iter_dict, file_name):
    """서버 iteration step별 kv_cache_usage 그래프 저장"""
    if not iter_dict:
        print("No kv usage data to plot")
        return
    
    iterations = []
    latencies = []
    
    for iteration_step in sorted(iter_dict.keys()):
        step_latencies = iter_dict[iteration_step]
        for latency in step_latencies:
            iterations.append(iteration_step)
            latencies.append(latency)
    
    if not iterations:
        print("No valid TPOT data to plot")
        return
    
    # 단일 그래프 생성 (선으로 연결)
    plt.figure(figsize=(12, 6))
    plt.plot(iterations, latencies, 'o-', markersize=4, linewidth=1, alpha=0.7, color='blue')
    plt.xlabel('Server Iteration Step')
    plt.ylabel('kv cache usage [ratio]')
    plt.title('kv cache usage by Iteration Step')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{file_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"kv plot saved to {file_name}.png")
    
    # 간단한 통계 정보 출력
    print(f"\nkv cache usage Statistics:")
    print(f"Total iteration steps executed: {len(iter_dict)}")  # 실제 실행된 iteration step 횟수
    print(f"Unique iteration step range: {min(iterations)} - {max(iterations)}")
    if latencies:
        print(f"Average kv cache usage: {np.mean(latencies):.4f}s")
        print(f"Min kv cache usage: {min(latencies):.4f}s")
        print(f"Max kv cache usage: {max(latencies):.4f}s")

def save_tpot_plots(iter_dict, file_name):
    """서버 iteration step별 token latency 그래프 저장"""
    if not iter_dict:
        print("No TPOT data to plot")
        return
    
    # iteration step과 해당 latency 데이터 준비
    iterations = []
    latencies = []
    
    for iteration_step in sorted(iter_dict.keys()):
        step_latencies = iter_dict[iteration_step]
        for latency in step_latencies:
            iterations.append(iteration_step)
            latencies.append(latency)
    
    if not iterations:
        print("No valid TPOT data to plot")
        return
    
    # 단일 그래프 생성 (선으로 연결)
    plt.figure(figsize=(12, 6))
    plt.plot(iterations, latencies, 'o-', markersize=4, linewidth=1, alpha=0.7, color='blue')
    plt.xlabel('Server Iteration Step')
    plt.ylabel('Token Latency [s]')
    plt.title('Token Latency by Iteration Step')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{file_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"TPOT plot saved to {file_name}.png")
    
    # 간단한 통계 정보 출출
    print(f"\nTPOT Statistics:")
    print(f"Total iteration steps executed: {len(iter_dict)}")  # 실제 실행된 iteration step 횟수
    print(f"Unique iteration step range: {min(iterations)} - {max(iterations)}")
    print(f"Total tokens processed: {len(latencies)}")  # 생성된 총 토큰 개수
    if latencies:
        print(f"Average token latency: {np.mean(latencies):.4f}s")
        print(f"Min token latency: {min(latencies):.4f}s")
        print(f"Max token latency: {max(latencies):.4f}s")

def create_performance_scatter_plots(results, middle_ratio=0.8):
    """
    TTFT vs TPOT 산포도 생성 및 성능 통계 계산
    
    Args:
        results: 실험 결과 리스트
        middle_ratio: 중간 구간 비율 (0.8이면 앞뒤 10%씩 제거하고 중간 80% 사용)
    """
    
    # 성공한 요청들의 데이터만 추출 (request_id 순서로 정렬)
    successful_results = [r for r in results if r['success']]
    successful_results.sort(key=lambda x: x['request_id'])  # request_id 순서로 정렬
    
    total_count = len(results)
    success_count = len(successful_results)
    
    print(f"Completed {success_count}/{total_count} requests successfully")
    
    if not successful_results:
        print("No successful results to plot or calculate stats")
        return {}, success_count
    
    # 중간 구간 계산
    if middle_ratio >= 1.0 or middle_ratio <= 0:
        print(f"Warning: Invalid middle_ratio {middle_ratio}, using all data")
        filtered_results = successful_results
        start_idx = 0
        end_idx = len(successful_results)
    else:
        total_successful = len(successful_results)
        skip_count = int(total_successful * (1 - middle_ratio) / 2)
        start_idx = skip_count
        end_idx = total_successful - skip_count
        filtered_results = successful_results[start_idx:end_idx]
    
    print(f"Using middle {middle_ratio*100}% of data: requests {start_idx} to {end_idx-1} (total: {len(filtered_results)})")
    
    if not filtered_results:
        print("No data remaining after filtering")
        return {}, success_count
    
    # 필터링된 데이터에서 통계 계산
    latency_values = [r['latency'] for r in filtered_results]
    ttft_values = [r['ttft'] for r in filtered_results if r['ttft'] > 0]
    tpot_values = [r['tpot'] for r in filtered_results if r['tpot'] > 0]
    
    # 통계 계산
    stats = {}
    stats['filtered_count'] = len(filtered_results)
    stats['filter_range'] = f"{start_idx}-{end_idx-1}"
    stats['avg_latency'] = np.mean(latency_values)
    
    if ttft_values:
        stats['avg_ttft'] = np.mean(ttft_values)
        stats['p90_ttft'] = np.percentile(ttft_values, 90)
        stats['p99_ttft'] = np.percentile(ttft_values, 99)
    
    if tpot_values:
        stats['avg_tpot'] = np.mean(tpot_values)
        stats['p90_tpot'] = np.percentile(tpot_values, 90)
        stats['p99_tpot'] = np.percentile(tpot_values, 99)
    
    # 통계 출력
    print(f"\n--- Performance Statistics (Middle {middle_ratio*100}% of requests) ---")
    print(f"Filtered data range: requests {start_idx} to {end_idx-1} ({len(filtered_results)} requests)")
    print(f"Average latency: {stats.get('avg_latency', 0):.4f}s")
    if 'avg_ttft' in stats:
        print(f"Average TTFT: {stats['avg_ttft']:.4f}s")
        print(f"P90 TTFT: {stats['p90_ttft']:.4f}s")
        print(f"P99 TTFT: {stats['p99_ttft']:.4f}s")
    if 'avg_tpot' in stats:
        print(f"Average TPOT: {stats['avg_tpot']:.4f}s")
        print(f"P90 TPOT: {stats['p90_tpot']:.4f}s")
        print(f"P99 TPOT: {stats['p99_tpot']:.4f}s")
    
    # 산포도를 위한 페어 데이터 (필터링된 결과만 사용)
    paired_data = [(r['tpot'], r['ttft']) for r in filtered_results 
                   if r['ttft'] > 0 and r['tpot'] > 0]  # x=TPOT, y=TTFT
    
    if not paired_data:
        print("No paired TTFT/TPOT data to plot")
        return stats, success_count
    
    paired_tpot, paired_ttft = zip(*paired_data)
    
    # 축 범위 계산 (여유있게)
    tpot_max = max(paired_tpot)
    ttft_max = max(paired_ttft)
    tpot_min = min(paired_tpot)
    ttft_min = min(paired_ttft)
    
    tpot_margin = (tpot_max - tpot_min) * 0.1  # 10% 여유
    ttft_margin = (ttft_max - ttft_min) * 0.1  # 10% 여유
    
    # 단일 산포도 생성
    plt.figure(figsize=(12, 8))  # 범례가 들어갈 공간을 위해 너비를 조금 늘림
    plt.scatter(paired_tpot, paired_ttft, alpha=0.7, s=50, color='blue', edgecolors='darkblue', linewidth=0.5)
    plt.xlabel('TPOT [s]')
    plt.ylabel('TTFT [s]')
    plt.title(f'TTFT vs TPOT Scatter Plot (Middle {middle_ratio*100}% of requests)')
    plt.grid(True, alpha=0.3)
    
    # 축 범위 설정 (여유있게)
    x_min, x_max = tpot_min - tpot_margin, tpot_max + tpot_margin
    y_min, y_max = ttft_min - ttft_margin, ttft_max + ttft_margin
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    # 평균값과 P90 값 계산
    avg_tpot = np.mean(paired_tpot)
    avg_ttft = np.mean(paired_ttft)
    p90_tpot = np.percentile(paired_tpot, 90)
    p90_ttft = np.percentile(paired_ttft, 90)
    
    # 평균선 추가 (실선)
    plt.axvline(avg_tpot, color='red', linestyle='-', alpha=0.8, linewidth=2, 
                label=f'Avg TPOT: {avg_tpot:.3f}s')
    plt.axhline(avg_ttft, color='green', linestyle='-', alpha=0.8, linewidth=2, 
                label=f'Avg TTFT: {avg_ttft:.3f}s')
    
    # P90 선 추가 (점선)
    plt.axvline(p90_tpot, color='red', linestyle='--', alpha=0.8, linewidth=2, 
                label=f'P90 TPOT: {p90_tpot:.3f}s')
    plt.axhline(p90_ttft, color='green', linestyle='--', alpha=0.8, linewidth=2, 
                label=f'P90 TTFT: {p90_ttft:.3f}s')
    
    # 범례 위치 조정 (그래프 밖으로 배치)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # 파일 저장
    scatter_file = f"performance_scatter_middle_{int(middle_ratio*100)}pct.png"
    plt.savefig(scatter_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Performance scatter plot saved to {scatter_file}")
    
    # 추가 통계 정보 출력
    print(f"\nDetailed Performance Statistics (Filtered Data):")
    print(f"TTFT - Mean: {np.mean(ttft_values):.4f}s, Std: {np.std(ttft_values):.4f}s")
    print(f"TPOT - Mean: {np.mean(tpot_values):.4f}s, Std: {np.std(tpot_values):.4f}s")
    if len(paired_data) > 1:
        correlation = np.corrcoef(paired_tpot, paired_ttft)[0, 1]
        print(f"TTFT-TPOT Correlation: {correlation:.4f}")
    
    return stats, success_count



async def run_throughput_test(
    api_url,
    model_name,
    input_len=512,
    output_len=32,
    num_requests=512,
    nestedfp=False,
):
    """
    Throughput Test: 모든 요청을 동시에 전송하여 offline serving E2E latency 측정
    """
    print(f"\n--- Running Throughput Test ---")
    print(f"Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Nested FP: {nestedfp}")
    print(f"  Input length: {input_len} tokens")
    print(f"  Output length: {output_len} tokens")
    print(f"  Number of requests: {num_requests}")

    # 전역 변수 초기화
    global iteration_details
    iteration_details.clear()

    # prompt 생성
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
        selected_words = random.choices(words, k=input_len * 2)
        prompt = " ".join(selected_words)
        encoded = tokenizer.encode(prompt, add_special_tokens=False)
        if len(encoded) > input_len:
            truncated_tokens = encoded[:input_len]
            prompt = tokenizer.decode(truncated_tokens)
        elif len(encoded) < input_len:
            while len(encoded) < input_len:
                additional_word = random.choice(words)
                test_prompt = prompt + " " + additional_word
                test_encoded = tokenizer.encode(test_prompt, add_special_tokens=False)
                if len(test_encoded) <= input_len:
                    prompt = test_prompt
                    encoded = test_encoded
                else:
                    break
        prompts.append(prompt)

    print(f"All prompts generated ({len(prompts)})")

    # 모든 요청을 동시에 전송
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

    # 통계 (파일 저장 없이 summary만 반환)
    success_count = sum(1 for r in results if r["success"])
    latencies = [r["latency"] for r in results if r["success"]]
    ttfts = [r["ttft"] for r in results if r["success"] and r["ttft"] > 0]
    tpots = [r["tpot"] for r in results if r["success"] and r["tpot"] > 0]

    total_tokens = sum(r["actual_generated_tokens"] for r in results if r["success"])
    throughput = total_tokens / total_experiment_time if total_experiment_time > 0 else 0

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

    # 파일 저장 없이 summary만 반환
    return summary


async def run_throughput_sweep(
    api_url,
    model_name,
    input_output_combinations=None,
    batch_sizes=[32, 64, 128, 256, 512],
    nestedfp=False,
):
    """
    여러 (input_len, output_len) 조합과 batch size에 대해 throughput test를 순차적으로 실행하고,
    각 결과(E2E latency, throughput 등)를 하나의 파일에 저장.
    
    Args:
        input_output_combinations: List of (input_len, output_len) tuples
                                   기본값: [(128,32), (256,32), (512,32), (1024,32),
                                            (128,512), (256,512), (512,512), (1024,512)]
    """
    # 기본 조합 설정
    if input_output_combinations is None:
        input_output_combinations = [
            (32, 512),
            (1024, 512),
            (32, 32),
            (1024, 32)
        ]
    
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
            
            # 결과에 input/output 정보 추가
            summary["input_len"] = input_len
            summary["output_len"] = output_len
            summary["batch_size"] = batch_size
            
            all_results.append(summary)

    # ✅ 모델명 기반 단일 sweep 파일 생성
    model_tag = os.path.basename(model_name.rstrip("/"))
    sweep_file = f"throughput_sweep_{model_tag}_{nestedfp}.json"

    # 구조화된 결과 저장
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

    # 요약 테이블 출력
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

async def main():
    parser = argparse.ArgumentParser(description="vLLM trace-based benchmark client")
    
    # Test mode 선택
    parser.add_argument("--test-mode",
                       choices=["trace", "throughput"],
                       default="trace",
                       help="Test mode: 'trace' for trace-based test, 'throughput' for throughput test")
    
    # Trace mode arguments
    parser.add_argument("--trace-file", 
                       default="./trace/azure_conv_0514_1400_20min_10.0x_tc.csv",
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
    
    # URL 검증 및 수정
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
                # 기본값: 8개의 조합
                combinations = [
                    (32, 32), (32, 512), (1024, 32), (1024, 512)
                ]
            
            # Parse batch sizes
            batch_sizes = [int(x) for x in args.throughput_batch_sizes.split(',')]
            
            # Throughput Test 실행
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