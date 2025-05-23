import asyncio
import argparse
import pandas as pd
import numpy as np
import time
from typing import List, Optional, Dict
from datetime import datetime
import io
import json
import os
import sys
import traceback
from dataclasses import dataclass, field
from tqdm.asyncio import tqdm
import aiohttp

# 기존 코드에서 정의한 데이터 클래스 (수정된 버전)
@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    model_name: Optional[str] = None
    logprobs: Optional[int] = None
    extra_body: Optional[dict] = None
    multi_modal_content: Optional[dict] = None
    ignore_eos: bool = False
    language: Optional[str] = None


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    output_tokens: int = 0
    ttft: float = 0.0  # Time to first token
    itl: list[float] = field(default_factory=list)  # list of inter-token latencies
    tpot: float = 0.0  # avg next-token latencies
    prompt_len: int = 0
    error: str = ""
    # 새로 추가: 토큰 도착 시간 기록
    token_arrival_times: list[float] = field(default_factory=list)
    request_sent_time: float = 0.0
    request_completed_time: float = 0.0

# OpenAI 호환 Completions API 요청 함수 (수정된 버전)
async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(
        ("completions", "profile")
    ), "OpenAI Completions API URL must end with 'completions' or 'profile'."

    AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
    async with aiohttp.ClientSession(trust_env=True,
                                     timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "model": request_func_input.model_name \
                if request_func_input.model_name else request_func_input.model,
            "prompt": request_func_input.prompt,
            "temperature": 0.0,
            "repetition_penalty": 1.0,
            "max_tokens": request_func_input.output_len,
            "logprobs": request_func_input.logprobs,
            "stream": True,
            "stream_options": {
                "include_usage": True,
            },
        }
        if request_func_input.ignore_eos:
            payload["ignore_eos"] = request_func_input.ignore_eos
        if request_func_input.extra_body:
            payload.update(request_func_input.extra_body)
        
        # vLLM 서버가 인증이 필요 없는 경우 빈 헤더로 설정
        headers = {}
        # OpenAI API를 사용하는 경우 API 키 설정
        openai_api_key = os.environ.get('OPENAI_API_KEY')
        if openai_api_key:
            headers["Authorization"] = f"Bearer {openai_api_key}"

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        st = time.perf_counter()
        output.request_sent_time = st  # 요청 전송 시간 기록
        most_recent_timestamp = st
        
        try:
            async with session.post(url=api_url, json=payload,
                                    headers=headers) as response:
                if response.status == 200:
                    first_chunk_received = False
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = chunk_bytes.decode("utf-8").removeprefix(
                            "data: ")
                        if chunk != "[DONE]":
                            data = json.loads(chunk)

                            # NOTE: Some completion API might have a last
                            # usage summary response without a token so we
                            # want to check a token was generated
                            if choices := data.get("choices"):
                                # Note that text could be empty here
                                # e.g. for special tokens
                                text = choices[0].get("text")
                                timestamp = time.perf_counter()
                                
                                # 각 토큰 도착 시간 기록 (빈 텍스트도 포함)
                                output.token_arrival_times.append(timestamp)
                                
                                # First token
                                if not first_chunk_received:
                                    first_chunk_received = True
                                    ttft = timestamp - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp -
                                                     most_recent_timestamp)

                                most_recent_timestamp = timestamp
                                generated_text += text or ""
                            elif usage := data.get("usage"):
                                output.output_tokens = usage.get(
                                    "completion_tokens")
                    
                    output.request_completed_time = time.perf_counter()  # 요청 완료 시간 기록
                    
                    if first_chunk_received:
                        output.success = True
                    else:
                        output.success = False
                        output.error = (
                            "Never received a valid chunk to calculate TTFT."
                            "This response will be marked as failed!")
                    output.generated_text = generated_text
                    output.latency = output.request_completed_time - st
                    
                    # TPOT 계산 추가
                    if output.itl:
                        output.tpot = sum(output.itl) / len(output.itl)
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output

# 임의의 프롬프트를 생성하는 함수
def generate_random_prompt(token_length):
    # "A"를 반복하여 정확한 토큰 길이 생성
    # 대부분의 토크나이저에서 "A"는 단일 토큰으로 처리됨
    return "A " * token_length

# 평균 request rate 계산 함수
def calculate_average_request_rate(trace_df):
    """trace 데이터에서 평균 request rate을 계산"""
    if len(trace_df) < 2:
        return 0.0
    
    # 첫 번째와 마지막 요청 간의 시간 간격
    total_duration = trace_df['relative_time'].max() - trace_df['relative_time'].min()
    
    if total_duration <= 0:
        return 0.0
    
    # 요청 간격들의 평균 계산
    time_diffs = trace_df['relative_time'].diff().dropna()
    if len(time_diffs) == 0:
        return 0.0
    
    avg_interval = time_diffs.mean()
    avg_rate = 1.0 / avg_interval if avg_interval > 0 else 0.0
    
    return avg_rate

# trace 파일에서 고정된 수의 요청 로드 (alpha 매개변수 추가)
def load_trace_data(file_path, alpha=1.0, num_requests=200):
    # CSV 파일에서 데이터 로드
    df = pd.read_csv(file_path)
    
    # 타임스탬프를 datetime 객체로 변환
    df['TIMESTAMP'] = pd.to_datetime(df.iloc[:, 0])
    
    # 열 이름 재설정 (첫 번째 열이 타임스탬프, 두 번째 열이 컨텍스트 토큰, 세 번째 열이 생성된 토큰)
    df.columns = ['TIMESTAMP', 'CONTEXT_TOKENS', 'GENERATED_TOKENS']
    
    # 고정된 수의 요청만 선택 (위에서부터)
    if len(df) > num_requests:
        df = df.head(num_requests)
        print(f"Using first {num_requests} requests from trace file")
    else:
        print(f"Warning: Only {len(df)} requests available, using all of them")
    
    # 첫 번째 타임스탬프를 기준으로 상대적 시간(초) 계산
    first_timestamp = df['TIMESTAMP'].min()
    # alpha 값을 적용하여 상대적 시간 조절
    # alpha < 1: 요청 간격이 늘어남 (느려짐), alpha > 1: 요청 간격이 줄어듦 (빨라짐)
    df['relative_time'] = (df['TIMESTAMP'] - first_timestamp).dt.total_seconds() / alpha
    
    # 평균 request rate 계산 및 출력
    original_rate = calculate_average_request_rate(df)
    adjusted_rate = original_rate * alpha  # alpha 적용된 rate
    
    print(f"Original trace request rate: {original_rate:.2f} req/s")
    print(f"Alpha-adjusted request rate: {adjusted_rate:.2f} req/s")
    
    return df

# 단일 요청 실행 함수 (수정된 버전)
async def execute_single_request(request_input, api_url, model_name, request_id):
    # RequestFuncInput 객체 생성
    input_obj = RequestFuncInput(
        prompt=generate_random_prompt(request_input['CONTEXT_TOKENS']),
        api_url=api_url,
        prompt_len=request_input['CONTEXT_TOKENS'],
        output_len=request_input['GENERATED_TOKENS'],
        model=model_name
    )
    
    # vLLM API 호출 (OpenAI 호환 방식)
    result = await async_request_openai_completions(input_obj)
    
    # request_id 추가
    return result, request_id

# trace 기반의 모든 요청 실행 (수정된 버전)
async def execute_trace_based_requests(trace_df, api_url, model_name):
    results = []
    start_time = time.time()
    
    print(f"Starting trace-based test with {len(trace_df)} requests")
    if len(trace_df) > 0:
        total_duration = trace_df['relative_time'].max()
        print(f"Estimated duration: {total_duration:.2f} seconds")
    
    for idx, row in trace_df.iterrows():
        # 현재 시간
        current_time = time.time() - start_time
        # trace 파일의 상대적 시간
        target_time = row['relative_time']
        
        # 필요한 경우 대기
        if current_time < target_time:
            await asyncio.sleep(target_time - current_time)
        
        # 비동기로 요청 전송 (request_id 추가)
        task = asyncio.create_task(execute_single_request(row, api_url, model_name, idx))
        results.append((task, row, idx))
        
        send_time = time.time() - start_time
        print(f"Request {idx+1}/{len(trace_df)} sent at {send_time:.2f}s: "
              f"Context tokens={row['CONTEXT_TOKENS']}, Generated tokens={row['GENERATED_TOKENS']}")
    
    print(f"All requests sent. Waiting for responses...")
    
    # 모든 요청 완료 대기 및 결과 수집
    final_results = []
    for task, row, request_id in results:
        result, _ = await task
        final_results.append({
            'request_id': request_id,
            'context_tokens': row['CONTEXT_TOKENS'],
            'generated_tokens': row['GENERATED_TOKENS'],
            'success': result.success,
            'latency': result.latency,
            'ttft': result.ttft,
            'tpot': result.tpot,
            'error': result.error,
            # 새로 추가된 필드들
            'request_sent_time': result.request_sent_time,
            'request_completed_time': result.request_completed_time,
            'token_arrival_times': result.token_arrival_times  # 모든 토큰 도착시간
        })
    
    total_duration = time.time() - start_time
    print(f"Experiment completed in {total_duration:.2f}s")
    
    return final_results

# 토큰 도착 시간 데이터를 CSV 형태로 변환
def convert_to_token_arrival_csv(results):
    """각 request별로 하나의 row, 토큰 도착 시간을 별도 컬럼으로"""
    rows = []
    max_tokens = max(len(r['token_arrival_times']) for r in results if r['success'])
    
    for result in results:
        if not result['success']:
            continue
            
        row = {
            'request_id': result['request_id'],
            'context_tokens': result['context_tokens'],
            'generated_tokens': result['generated_tokens'],
            'request_sent_time': result['request_sent_time'],
            'request_completed_time': result['request_completed_time'],
            'total_latency': result['latency'],
            'ttft': result['ttft'],
            'tpot': result['tpot']
        }
        
        # 토큰 도착 시간들을 별도 컬럼으로 추가
        for i, arrival_time in enumerate(result['token_arrival_times']):
            row[f'token_{i+1}_arrival_time'] = arrival_time
        
        # 나머지 토큰 컬럼은 빈 값으로 채움
        for i in range(len(result['token_arrival_times']), max_tokens):
            row[f'token_{i+1}_arrival_time'] = None
            
        rows.append(row)
    
    return pd.DataFrame(rows)

# 각 토큰별 상세 정보를 위한 테이블 생성 (시간대별 P90 ITL 분석용)
def create_token_level_table(results):
    """각 토큰별로 하나의 row를 생성 (ITL 포함)"""
    token_rows = []
    
    for result in results:
        if not result['success'] or len(result['token_arrival_times']) == 0:
            continue
            
        # 첫 번째 토큰은 ITL이 없음 (TTFT만 존재)
        token_rows.append({
            'request_id': result['request_id'],
            'token_index': 0,
            'token_arrival_time': result['token_arrival_times'][0],
            'itl': None,  # 첫 번째 토큰은 ITL이 없음
            'cumulative_tokens': 1,
            'is_first_token': True
        })
        
        # 나머지 토큰들의 ITL 계산
        for i in range(1, len(result['token_arrival_times'])):
            itl = result['token_arrival_times'][i] - result['token_arrival_times'][i-1]
            token_rows.append({
                'request_id': result['request_id'],
                'token_index': i,
                'token_arrival_time': result['token_arrival_times'][i],
                'itl': itl,
                'cumulative_tokens': i + 1,
                'is_first_token': False
            })
    
    return pd.DataFrame(token_rows)

# 시간대별 P90 ITL 계산
def calculate_p90_itl_over_time(token_df, interval_seconds=1.0):
    """시간대별 P90 ITL 계산"""
    if token_df.empty:
        return pd.DataFrame()
    
    # 첫 번째 토큰은 제외 (ITL이 None)
    itl_tokens = token_df[token_df['itl'].notna()].copy()
    
    if itl_tokens.empty:
        return pd.DataFrame()
    
    # 기준 시간 (가장 빠른 토큰 도착 시간)
    min_time = itl_tokens['token_arrival_time'].min()
    max_time = itl_tokens['token_arrival_time'].max()
    
    # 시간 구간 생성
    time_bins = np.arange(min_time, max_time + interval_seconds, interval_seconds)
    
    # 각 토큰을 시간 구간에 할당
    itl_tokens['time_bin'] = pd.cut(itl_tokens['token_arrival_time'], bins=time_bins, right=False)
    
    # 각 시간 구간별 P90 ITL 계산
    p90_results = []
    
    for bin_label, group in itl_tokens.groupby('time_bin'):
        if group.empty:
            continue
            
        bin_start = bin_label.left
        bin_end = bin_label.right
        itl_values = group['itl'].values
        
        if len(itl_values) > 0:
            p90_itl = np.percentile(itl_values, 90)
            p50_itl = np.percentile(itl_values, 50)
            avg_itl = np.mean(itl_values)
            token_count = len(itl_values)
            request_count = group['request_id'].nunique()
            
            p90_results.append({
                'time_interval_start': bin_start,
                'time_interval_end': bin_end,
                'time_interval_center': (bin_start + bin_end) / 2,
                'p90_itl': p90_itl,
                'p50_itl': p50_itl,
                'avg_itl': avg_itl,
                'token_count': token_count,
                'request_count': request_count
            })
    
    return pd.DataFrame(p90_results)

# 시간대별 Request Rate 계산 (새로 추가)
def calculate_request_rate_over_time(results, interval_seconds=1.0):
    """시간대별 Request Rate 계산"""
    if not results:
        return pd.DataFrame()
    
    # 성공한 요청들의 전송 시간 추출
    request_times = []
    for result in results:
        if result['success']:
            request_times.append({
                'request_id': result['request_id'],
                'request_sent_time': result['request_sent_time']
            })
    
    if not request_times:
        return pd.DataFrame()
    
    request_df = pd.DataFrame(request_times)
    
    # 기준 시간 (가장 빠른 요청 전송 시간)
    min_time = request_df['request_sent_time'].min()
    max_time = request_df['request_sent_time'].max()
    
    # 시간 구간 생성
    time_bins = np.arange(min_time, max_time + interval_seconds, interval_seconds)
    
    # 각 요청을 시간 구간에 할당
    request_df['time_bin'] = pd.cut(request_df['request_sent_time'], bins=time_bins, right=False)
    
    # 각 시간 구간별 요청 개수 계산
    rate_results = []
    
    for bin_label, group in request_df.groupby('time_bin'):
        if group.empty:
            continue
            
        bin_start = bin_label.left
        bin_end = bin_label.right
        request_count = len(group)
        request_rate = request_count / interval_seconds  # 초당 요청 수
        
        rate_results.append({
            'time_interval_start': bin_start,
            'time_interval_end': bin_end,
            'time_interval_center': (bin_start + bin_end) / 2,
            'request_count': request_count,
            'request_rate': request_rate,
            'interval_seconds': interval_seconds
        })
    
    return pd.DataFrame(rate_results)

# 성능 통계 계산 함수 (기존과 동일)
def calculate_performance_stats(results):
    success_count = sum(1 for r in results if r['success'])
    
    stats = {}
    
    if success_count > 0:
        # 지연 시간 통계
        latency_values = [r['latency'] for r in results if r['success']]
        stats['avg_latency'] = np.mean(latency_values)
        
        # TTFT 통계
        ttft_values = [r['ttft'] for r in results if r['success'] and r['ttft'] > 0]
        if ttft_values:
            stats['avg_ttft'] = np.mean(ttft_values)
            stats['p25_ttft'] = np.percentile(ttft_values, 25)
            stats['p50_ttft'] = np.percentile(ttft_values, 50)
            stats['p75_ttft'] = np.percentile(ttft_values, 75)
            stats['p90_ttft'] = np.percentile(ttft_values, 90)
            stats['p99_ttft'] = np.percentile(ttft_values, 99)
        
        # TPOT 통계
        tpot_values = [r['tpot'] for r in results if r['success'] and r['tpot'] > 0]
        if tpot_values:
            stats['avg_tpot'] = np.mean(tpot_values)
            stats['p25_tpot'] = np.percentile(tpot_values, 25)
            stats['p50_tpot'] = np.percentile(tpot_values, 50)
            stats['p75_tpot'] = np.percentile(tpot_values, 75)
            stats['p90_tpot'] = np.percentile(tpot_values, 90)
            stats['p99_tpot'] = np.percentile(tpot_values, 99)
    
    return stats, success_count

# 단일 alpha 값에 대한 실험 실행 (수정된 버전)
async def run_experiment_with_alpha(alpha, trace_file, api_url, model_name, num_requests=200, p90_interval=1.0):
    print(f"\n--- Running experiment with alpha={alpha:.2f} ---")
    print(f"Loading {num_requests} requests from trace data with rate multiplier alpha={alpha:.2f}")
    trace_df = load_trace_data(trace_file, alpha, num_requests)
    print(f"Loaded {len(trace_df)} requests from trace file")
    
    # trace 기반 요청 실행
    results = await execute_trace_based_requests(trace_df, api_url, model_name)
    
    # 성능 통계 계산
    stats, success_count = calculate_performance_stats(results)
    print(f"Completed {success_count}/{len(results)} requests successfully")
    
    # 실제 달성된 request rate 계산
    if results:
        experiment_duration = max(r['request_completed_time'] for r in results if r['success']) - min(r['request_sent_time'] for r in results if r['success']) if success_count > 0 else 0
        actual_rate = len(results) / experiment_duration if experiment_duration > 0 else 0
        print(f"Actual achieved request rate: {actual_rate:.2f} req/s")
    
    # 성능 통계 출력
    if stats:
        print(f"Average latency: {stats.get('avg_latency', 0):.4f}s")
        
        if 'avg_ttft' in stats:
            print("\nTTFT Statistics:")
            print(f"Average TTFT: {stats['avg_ttft']:.4f}s")
            print(f"P25 TTFT: {stats['p25_ttft']:.4f}s")
            print(f"P50 TTFT: {stats['p50_ttft']:.4f}s")
            print(f"P75 TTFT: {stats['p75_ttft']:.4f}s")
            print(f"P90 TTFT: {stats['p90_ttft']:.4f}s")
            print(f"P99 TTFT: {stats['p99_ttft']:.4f}s")
        
        if 'avg_tpot' in stats:
            print("\nTPOT Statistics:")
            print(f"Average TPOT: {stats['avg_tpot']:.4f}s")
            print(f"P25 TPOT: {stats['p25_tpot']:.4f}s")
            print(f"P50 TPOT: {stats['p50_tpot']:.4f}s")
            print(f"P75 TPOT: {stats['p75_tpot']:.4f}s")
            print(f"P90 TPOT: {stats['p90_tpot']:.4f}s")
            print(f"P99 TPOT: {stats['p99_tpot']:.4f}s")
    else:
        print("No valid statistics available")
    
    # 파일 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 토큰 도착 시간 CSV 저장
    token_arrival_df = convert_to_token_arrival_csv(results)
    token_csv_file = f"token_arrivals_alpha{alpha:.2f}_{timestamp}.csv"
    token_arrival_df.to_csv(token_csv_file, index=False)
    print(f"\nToken arrival times saved to {token_csv_file}")
    print(f"Shape: {token_arrival_df.shape}")
    
    # 2. 토큰별 상세 데이터 생성 및 저장
    token_level_df = create_token_level_table(results)
    token_level_file = f"token_level_data_alpha{alpha:.2f}_{timestamp}.csv"
    token_level_df.to_csv(token_level_file, index=False)
    print(f"Token-level data saved to {token_level_file}")
    print(f"Shape: {token_level_df.shape}")
    
    # 3. 시간대별 P90 ITL 계산 및 저장
    p90_itl_df = calculate_p90_itl_over_time(token_level_df, interval_seconds=p90_interval)
    p90_file = f"p90_itl_over_time_alpha{alpha:.2f}_{timestamp}.csv"
    p90_itl_df.to_csv(p90_file, index=False)
    print(f"P90 ITL over time saved to {p90_file}")
    print(f"Shape: {p90_itl_df.shape}")
    print(f"Time interval: {p90_interval} seconds")
    
    # 4. 시간대별 Request Rate 계산 및 저장 (새로 추가)
    request_rate_df = calculate_request_rate_over_time(results, interval_seconds=p90_interval)
    request_rate_file = f"request_rate_count_alpha{alpha:.2f}_{timestamp}.csv"
    request_rate_df.to_csv(request_rate_file, index=False)
    print(f"Request rate over time saved to {request_rate_file}")
    print(f"Shape: {request_rate_df.shape}")
    
    # 5. 기존 형태의 결과 파일도 저장
    results_file = f"benchmark_results_alpha{alpha:.2f}_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to {results_file}")
    
    return alpha, stats, success_count, len(results)

# 여러 알파 값에 대한 실험 실행 (기존과 동일)
async def run_experiments(alphas, trace_file, api_url, model_name, num_requests=200):
    summary_results = []
    
    # 첫 번째 실험 전에 원본 trace의 평균 request rate 계산
    print("Loading trace file to calculate original request rate...")
    original_df = pd.read_csv(trace_file)
    original_df['TIMESTAMP'] = pd.to_datetime(original_df.iloc[:, 0])
    original_df.columns = ['TIMESTAMP', 'CONTEXT_TOKENS', 'GENERATED_TOKENS']
    
    if len(original_df) > num_requests:
        original_df = original_df.head(num_requests)
    
    first_timestamp = original_df['TIMESTAMP'].min()
    original_df['relative_time'] = (original_df['TIMESTAMP'] - first_timestamp).dt.total_seconds()
    original_rate = calculate_average_request_rate(original_df)
    
    print(f"\nOriginal trace statistics:")
    print(f"Total number of requests: {len(original_df)}")
    print(f"Time span: {original_df['relative_time'].max():.2f} seconds")
    print(f"Average request rate: {original_rate:.2f} req/s")
    print(f"Using {num_requests} requests for experiments\n")
    
    for alpha in alphas:
        alpha_result = await run_experiment_with_alpha(alpha, trace_file, api_url, model_name, num_requests)
        summary_results.append(alpha_result)
    
    # 전체 결과 요약 CSV 저장
    summary_df = []
    for alpha, stats, success_count, total_count in summary_results:
        if stats:
            adjusted_rate = original_rate * alpha
            row = {
                'alpha': alpha, 
                'target_rate': adjusted_rate,
                'success_rate': success_count / total_count,
                'total_requests': total_count
            }
            row.update(stats)
            summary_df.append(row)
    
    if summary_df:
        summary_df = pd.DataFrame(summary_df)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = f"benchmark_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\nSummary results saved to {summary_file}")
    
    return summary_df

# 메인 함수 (기존과 동일)
async def main():
    parser = argparse.ArgumentParser(description="vLLM trace-based benchmark client (token arrival tracking)")
    parser.add_argument("--trace-file", 
                       default="/disk/dualfp_vllm_test/end_to_end_test/online/trace/trace_0510.csv",
                       help="Path to trace CSV file")
    parser.add_argument("--api-url", 
                       default="http://0.0.0.0:8000/v1/completions", 
                       help="vLLM server API URL")
    parser.add_argument("--model", 
                       default="/disk/models/Mistral-Small-24B-Base-2501",
                       help="Model name or path")
    parser.add_argument("--alpha", 
                       type=float, 
                       default=1,
                       help="Request rate multiplier. If not provided, will run multiple alphas")
    parser.add_argument("--num-requests", 
                       type=int, 
                       default=5000,
                       help="Number of requests to use for the experiment")
    args = parser.parse_args()
    
    # API URL이 completions으로 끝나는지 확인하고, 필요한 경우 수정
    if not args.api_url.endswith(("completions", "profile")):
        if not args.api_url.endswith('/'):
            args.api_url += '/v1/completions'
        else:
            args.api_url += 'v1/completions'
    
    # 단일 알파 또는 여러 알파 값으로 실험 실행
    if args.alpha is not None:
        await run_experiment_with_alpha(args.alpha, args.trace_file, args.api_url, args.model, args.num_requests)
    else:
        alphas = [0.04, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36, 0.4]
        summary_df = await run_experiments(alphas, args.trace_file, args.api_url, args.model, args.num_requests)

# 스크립트 실행
if __name__ == "__main__":
    asyncio.run(main())