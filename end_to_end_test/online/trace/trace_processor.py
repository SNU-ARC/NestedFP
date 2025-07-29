import pandas as pd
import argparse
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
import os



def clip_by_day(input_file, output_file, start_time_str, end_time_str):
    df = pd.read_csv(input_file)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='mixed', utc=True)

    start_time = pd.Timestamp(start_time_str, tz='UTC')
    end_time = pd.Timestamp(end_time_str, tz='UTC')

    filtered_df = df[(df['TIMESTAMP'] >= start_time) & (df['TIMESTAMP'] <= end_time)]
    filtered_df.to_csv(output_file, index=False)

    print(f"[clip_by_day] Saved {len(filtered_df)} rows → {output_file}")

def extend_request_rate(input_file, output_file, n_copies, time_offset_us):
    df = pd.read_csv(input_file)

    expanded_rows = []
    time_offset = timedelta(microseconds=time_offset_us)

    for idx, row in df.iterrows():
        base_time = pd.to_datetime(row['TIMESTAMP'])
        for i in range(n_copies):
            new_row = row.copy()
            new_row['TIMESTAMP'] = base_time + i * time_offset
            expanded_rows.append(new_row)

    expanded_df = pd.DataFrame(expanded_rows)
    expanded_df.to_csv(output_file, index=False)

    print(f"[extend_request_rate] Expanded {len(df)} rows → {len(expanded_df)} rows → {output_file}")


def scale_request_rate(input_file, output_file, scale_factor, seed=42):
    """
    확률적 샘플링을 통해 요청 비율을 조절합니다.
    시간 분포 패턴은 유지하면서 전체 요청 수만 줄입니다.
    
    Args:
        input_file: 입력 CSV 파일
        output_file: 출력 CSV 파일
        scale_factor: 스케일 팩터 (0.2 = 20%만 유지, 0.5 = 50%만 유지)
        seed: 랜덤 시드 (재현성을 위해)
    """
    # 랜덤 시드 설정
    np.random.seed(seed)
    
    # 파일 로드
    print(f"파일 '{input_file}' 로드 중...")
    df = pd.read_csv(input_file)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='mixed', utc=True)
    
    # 원본 정보
    original_count = len(df)
    original_duration = (df['TIMESTAMP'].max() - df['TIMESTAMP'].min()).total_seconds()
    original_rate = original_count / original_duration
    
    print(f"원본 요청 수: {original_count}")
    print(f"원본 기간: {original_duration:.2f} 초")
    print(f"원본 평균 요청 비율: {original_rate:.2f} requests/second")
    
    # 정확한 개수 유지 방식으로 샘플링 (더 안정적)
    n_keep = int(original_count * scale_factor)
    if n_keep == 0:
        n_keep = 1  # 최소 1개는 유지
    
    # 시간 순서를 유지하면서 랜덤 선택
    keep_indices = np.random.choice(original_count, n_keep, replace=False)
    keep_indices = np.sort(keep_indices)
    df_scaled = df.iloc[keep_indices].copy()
    
    print(f"선택된 요청 수: {n_keep} / {original_count} ({n_keep/original_count*100:.1f}%)")
    
    # 인덱스 리셋
    df_scaled.reset_index(drop=True, inplace=True)
    
    # 결과 저장
    df_scaled.to_csv(output_file, index=False)
    
    # 결과 정보 출력
    scaled_count = len(df_scaled)
    scaled_duration = original_duration  # 시간 구간은 동일
    scaled_rate = scaled_count / scaled_duration
    
    print(f"\n=== 스케일링 결과 ===")
    print(f"[scale_request_rate] Scale factor: {scale_factor}")
    print(f"[scale_request_rate] Original requests: {original_count}")
    print(f"[scale_request_rate] Scaled requests: {scaled_count}")
    print(f"[scale_request_rate] New duration: {scaled_duration:.2f} seconds")
    print(f"[scale_request_rate] Average rate: {scaled_rate:.2f} requests/second")
    print(f"[scale_request_rate] Saved to: {output_file}")
    
    return df_scaled


def clip_rows(input_file, output_file, n_rows):
    df = pd.read_csv(input_file)
    df_short = df.iloc[:n_rows]
    df_short.to_csv(output_file, index=False)

    print(f"[clip_rows] Saved first {len(df_short)} rows → {output_file}")
    

def plot_request_rate(input_file, bin_interval_ms=50, moving_avg_window_s=0.5, figsize=(12,6)):
    """
    input_file: 요청 arrival time이 기록된 CSV
    bin_interval_ms: binning 간격 (ex. 50ms = 0.05초)
    moving_avg_window_s: 몇 초 구간을 평균내서 smoothing할지 (ex. 0.5초)
    """
    df = pd.read_csv(input_file)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], utc=True)

    # Relative time 계산 (초 단위)
    first_timestamp = df['TIMESTAMP'].min()
    df['relative_time'] = (df['TIMESTAMP'] - first_timestamp).dt.total_seconds()

    relative_times = df['relative_time'].values

    # bin_interval 설정
    bin_interval_s = bin_interval_ms / 1000.0  # ms → s
    max_time = relative_times.max()
    bins = np.arange(0, max_time + bin_interval_s, bin_interval_s)

    # Histogram binning
    request_counts, _ = np.histogram(relative_times, bins=bins)
    bin_centers = bins[:-1] + bin_interval_s / 2

    # Moving average window (초 단위 → 몇 개 bin을 평균?)
    moving_avg_window_bins = int(moving_avg_window_s / bin_interval_s)
    if moving_avg_window_bins < 1:
        moving_avg_window_bins = 1

    request_counts_smoothed = np.convolve(request_counts, np.ones(moving_avg_window_bins)/moving_avg_window_bins, mode='same')

    # Plot
    plt.figure(figsize=figsize)
    plt.plot(bin_centers, request_counts_smoothed, linewidth=1)
    plt.xlabel("Relative Time (s)")
    plt.ylabel(f"Request Rate (requests per {bin_interval_ms}ms)")
    plt.title(f"Request Rate over Time ({bin_interval_ms}ms binning, {moving_avg_window_s:.1f}s Moving Avg)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../figures/request_rate_plot.png", dpi=300)
    print("[plot_request_rate] Saved figure to ../figures/request_rate_plot.png")
import random

def extend_high_rate(input_file, output_file, rate_multiplier=3, duration_seconds=None):
    df = pd.read_csv(input_file)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], utc=True)

    if duration_seconds is None:
        duration_seconds = (df['TIMESTAMP'].max() - df['TIMESTAMP'].min()).total_seconds()

    first_timestamp = df['TIMESTAMP'].min()
    last_timestamp = df['TIMESTAMP'].max()

    total_requests = len(df)
    avg_request_rate = total_requests / duration_seconds
    new_request_rate = avg_request_rate * rate_multiplier
    n_new_requests = int(new_request_rate * duration_seconds)
    interval_between_requests = timedelta(seconds=duration_seconds / n_new_requests)

    # 통계 기반 생성
    mean_context_tokens = 1631.58
    std_context_tokens = 300
    mean_ratio = 0.1039
    std_ratio = 0.02

    min_context_tokens = 1
    max_context_tokens = 7999
    min_generated_tokens = 1
    max_generated_tokens = 1500

    expected_columns = df.columns.tolist()

    new_rows = []
    for i in range(n_new_requests):
        new_time = last_timestamp + (i+1) * interval_between_requests

        # ContextTokens 생성
        context_tokens = int(np.clip(np.random.normal(loc=mean_context_tokens, scale=std_context_tokens), min_context_tokens, max_context_tokens))

        # GeneratedTokens 생성 (ContextTokens에 비례하되 약간 랜덤성)
        ratio = np.clip(np.random.normal(loc=mean_ratio, scale=std_ratio), 0.05, 0.5)
        generated_tokens = int(np.clip(context_tokens * ratio, min_generated_tokens, max_generated_tokens))

        new_row = {
            'TIMESTAMP': new_time,
            'ContextTokens': context_tokens,
            'GeneratedTokens': generated_tokens
        }

        for col in expected_columns:
            if col not in new_row:
                new_row[col] = np.nan

        new_rows.append(new_row)

    df_new = pd.DataFrame(new_rows, columns=expected_columns)

    # 앞 df + 뒤 df 합치기
    df_combined = pd.concat([df, df_new], ignore_index=True)

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_combined.to_csv(output_file, index=False)

    print(f"[extend_high_rate] Appended {n_new_requests} new requests → {output_file}")

def debug_per_second_stats(df):
    per_second = df.resample("1S").size()
    print("=== Per-Second Request Rate Stats ===")
    print(per_second.describe())
    print("\nMax request rate occurs at:", per_second.idxmax(), "→", per_second.max())
    print("\nValue counts (sorted):")
    print(per_second.value_counts().sort_index())
    print("======================================")


def analyze_variability(input_file):
    import pandas as pd

    def debug_per_second_stats(df):
        per_second = df.resample("1S").size()
        print("=== Per-Second Request Rate Stats ===")
        print(per_second.describe())
        print("\nMax request rate occurs at:", per_second.idxmax(), "→", per_second.max())
        print("\nValue counts (sorted):")
        print(per_second.value_counts().sort_index())
        print("======================================")

    def find_max_variability_window(df, freq='1H'):
        df_resampled = df.resample("1S").size()
        max_diff = -1
        max_window_start = None
        max_window_values = None

        for window_start, window_df in df_resampled.resample(freq):
            if len(window_df) == 0:
                continue
            diff = window_df.max() - window_df.min()
            if diff > max_diff:
                max_diff = diff
                max_window_start = window_start
                max_window_values = window_df

        print(f"\n[Max Variability Window - {freq}]")
        print(f"→ Start: {max_window_start}")
        print(f"→ Max rate: {max_window_values.max()}")
        print(f"→ Min rate: {max_window_values.min()}")
        print(f"→ Diff: {max_diff}")
        return max_window_start, max_window_values

    # --- Load input
    df = pd.read_csv(input_file)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], utc=True)
    df.set_index('TIMESTAMP', inplace=True)

    # --- Debug stats
    debug_per_second_stats(df)

    # --- Global full-trace stats
    per_second = df.resample("1S").size()
    full_trace_diff = per_second.max() - per_second.min()

    # --- Per-minute variability
    per_minute_diff = []
    for _, window_df in df.resample("1min"):
        if len(window_df) == 0:
            continue
        counts = window_df.resample("1S").size()
        per_minute_diff.append(counts.max() - counts.min())
    per_minute_max_diff = max(per_minute_diff) if per_minute_diff else 0

    # --- Per-hour variability
    per_hour_diff = []
    for _, window_df in df.resample("1H"):
        if len(window_df) == 0:
            continue
        counts = window_df.resample("1S").size()
        per_hour_diff.append(counts.max() - counts.min())
    per_hour_max_diff = max(per_hour_diff) if per_hour_diff else 0

    print("\n[analyze_variability] Max (max - min) request rate differences:")
    print(f"  • Per Minute : {per_minute_max_diff}")
    print(f"  • Per Hour   : {per_hour_max_diff}")
    print(f"  • Entire Trace (1s bins) : {full_trace_diff}")

    # --- 추가: 최대 변동 구간 상세 정보 출력
    find_max_variability_window(df, freq='1H')
    find_max_variability_window(df, freq='1min')




import pandas as pd
import matplotlib.pyplot as plt

def plot_request_rate_raw(input_file, output_file="request_rate.pdf"):
    # Load and parse
    df = pd.read_csv(input_file)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], utc=True)
    df.set_index('TIMESTAMP', inplace=True)

    # Resample to 1-second bins (raw count)
    per_second = df.resample("1S").size()
    avg_rate = per_second.mean()

    # Set global font sizes
    plt.rcParams.update({
        'font.size': 12,         # 기본 폰트
        'axes.titlesize': 18,    # 제목
        'axes.labelsize': 20,    # x, y축 레이블
        'xtick.labelsize': 16,   # x축 눈금
        'ytick.labelsize': 18,   # y축 눈금
        'legend.fontsize': 18    # 범례
    })

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(per_second.index, per_second.values, color='blue', linewidth=0.6, label='Raw Request Rate (1s)')
    plt.axhline(y=avg_rate, color='red', linestyle='--', linewidth=1.2, label='Average Request Rate')

    # Labels and formatting
    plt.xlabel("Time")
    plt.ylabel("Request Rate (reqs/s)")
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    # Save
    plt.savefig(output_file, dpi=400)
    print(f"[plot_request_rate_raw] Saved figure to {output_file}")




    
def main():
    parser = argparse.ArgumentParser(...)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str)
    parser.add_argument('--start_time', type=str)
    parser.add_argument('--end_time', type=str)
    parser.add_argument('--n_copies', type=int)
    parser.add_argument('--time_offset_us', type=int)
    parser.add_argument('--n_rows', type=int)
    parser.add_argument('--rate_multiplier', type=int, default=3, help="Rate multiplier for extend_high_rate task")
    parser.add_argument('--duration_seconds', type=int, help="Duration for high rate extension (optional)")
    parser.add_argument('--scale_factor', type=float, help="Scale factor for scale_request_rate task (e.g., 0.2, 0.4)")
    parser.add_argument('--bin_interval_ms', type=int, default=1000, help="Bin interval for plot_request_rate")
    parser.add_argument('--moving_avg_window_s', type=float, default=10, help="Moving average window size in seconds for plot_request_rate")

    parser.add_argument('--task', type=str, required=True, choices=['clip_day', 'extend_rate', 'clip_rows', 'plot_request_rate', 'extend_high_rate', 'analyze_variability', 'plot_request_rate_raw', 'scale_request_rate'],
                    help="Task to perform...")

    args = parser.parse_args()

    if args.task == 'clip_day':
        if args.start_time is None or args.end_time is None:
            raise ValueError("start_time and end_time must be specified for clip_day task.")
        clip_by_day(args.input, args.output, args.start_time, args.end_time)
    
    elif args.task == 'extend_rate':
        n_copies = args.n_copies if args.n_copies else 10
        time_offset_us = args.time_offset_us if args.time_offset_us else 10
        extend_request_rate(args.input, args.output, n_copies, time_offset_us)

    elif args.task == 'clip_rows':
        n_rows = args.n_rows if args.n_rows else 2000
        clip_rows(args.input, args.output, n_rows)

    elif args.task == 'plot_request_rate':
        plot_request_rate(args.input)
    
    elif args.task == 'extend_high_rate':
        extend_high_rate(args.input, args.output, rate_multiplier=args.rate_multiplier, duration_seconds=args.duration_seconds)
        
    elif args.task == 'analyze_variability':
        analyze_variability(args.input)
    
    elif args.task == 'plot_request_rate_raw':
        plot_request_rate_raw(args.input)
        
    elif args.task == 'scale_request_rate':
        if args.scale_factor is None:
            raise ValueError("scale_factor must be specified for scale_request_rate task.")
        scale_request_rate(args.input, args.output, args.scale_factor)

if __name__ == "__main__":
    main()