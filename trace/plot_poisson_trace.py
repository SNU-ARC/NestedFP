import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from datetime import datetime, timedelta
import os


def generate_poisson_trace(input_file, output_file):
    """
    기존 trace 데이터를 바탕으로 Poisson process로 새로운 timestamp를 생성합니다.
    
    Args:
        input_file: 입력 CSV 파일
        output_file: 출력 CSV 파일
    """
    print(f"\n[Poisson Trace Generation] 시작")
    print(f"입력 파일: {input_file}")
    
    # 파일 로드
    df = pd.read_csv(input_file)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='mixed', utc=True)
    
    # 기본 정보 추출
    original_count = len(df)
    first_time = df['TIMESTAMP'].iloc[0]
    last_time = df['TIMESTAMP'].iloc[-1]
    total_duration = (last_time - first_time).total_seconds()
    original_rate = (original_count - 1) / total_duration if total_duration > 0 else 0
    
    print(f"원본 통계:")
    print(f"  - 총 요청 수: {original_count}")
    print(f"  - 첫 번째 요청: {first_time}")
    print(f"  - 마지막 요청: {last_time}")
    print(f"  - 총 시간: {total_duration:.2f}초")
    print(f"  - 평균 요청 비율: {original_rate:.4f} reqs/s")
    
    # 새로운 DataFrame 생성 (ContextTokens, GeneratedTokens 등 다른 컬럼들은 유지)
    df_poisson = df.copy()
    
    if original_count <= 1:
        print("요청이 1개 이하입니다. Poisson process 생성을 건너뜁니다.")
        df_poisson.to_csv(output_file, index=False)
        return df_poisson
    
    # Poisson process로 새로운 timestamp 생성
    print(f"\nPoisson process 생성 중...")
    print(f"Lambda (rate): {original_rate:.4f}")
    
    # Exponential distribution으로 inter-arrival time 생성
    # lambda = rate이므로, scale = 1/rate
    scale = 1.0 / original_rate if original_rate > 0 else 1.0
    inter_arrival_times = np.random.exponential(scale=scale, size=original_count-1)
    
    print(f"생성된 inter-arrival times 통계:")
    print(f"  - 평균: {np.mean(inter_arrival_times):.4f}초")
    print(f"  - 표준편차: {np.std(inter_arrival_times):.4f}초")
    print(f"  - 합계: {np.sum(inter_arrival_times):.2f}초")
    
    # 생성된 inter-arrival time들을 전체 시간에 맞게 normalize
    current_total = np.sum(inter_arrival_times)
    scaling_factor = total_duration / current_total
    inter_arrival_times_scaled = inter_arrival_times * scaling_factor
    
    print(f"Scaling factor: {scaling_factor:.6f}")
    print(f"Scaled inter-arrival times 합계: {np.sum(inter_arrival_times_scaled):.2f}초")
    
    # 새로운 timestamp 계산
    new_timestamps = [first_time]  # 첫 번째는 기존과 동일
    
    current_time = first_time
    for interval in inter_arrival_times_scaled:
        current_time += timedelta(seconds=interval)
        new_timestamps.append(current_time)
    
    # 마지막 timestamp가 정확히 맞는지 확인 (부동소수점 오차 때문에 약간의 차이가 있을 수 있음)
    if len(new_timestamps) > 1:
        new_timestamps[-1] = last_time  # 마지막을 정확히 맞춤
    
    # DataFrame에 새로운 timestamp 적용
    df_poisson['TIMESTAMP'] = new_timestamps
    
    # 결과 통계
    new_duration = (df_poisson['TIMESTAMP'].max() - df_poisson['TIMESTAMP'].min()).total_seconds()
    new_rate = (original_count - 1) / new_duration if new_duration > 0 else 0
    
    print(f"\n생성된 Poisson trace 통계:")
    print(f"  - 총 요청 수: {len(df_poisson)} (원본과 동일)")
    print(f"  - 첫 번째 요청: {df_poisson['TIMESTAMP'].iloc[0]}")
    print(f"  - 마지막 요청: {df_poisson['TIMESTAMP'].iloc[-1]}")
    print(f"  - 총 시간: {new_duration:.2f}초")
    print(f"  - 평균 요청 비율: {new_rate:.4f} reqs/s")
    
    # Inter-arrival time 분석
    if len(df_poisson) > 1:
        new_intervals = []
        for i in range(1, len(df_poisson)):
            interval = (df_poisson['TIMESTAMP'].iloc[i] - df_poisson['TIMESTAMP'].iloc[i-1]).total_seconds()
            new_intervals.append(interval)
        
        print(f"  - 새로운 inter-arrival times 평균: {np.mean(new_intervals):.4f}초")
        print(f"  - 새로운 inter-arrival times 표준편차: {np.std(new_intervals):.4f}초")
    
    # 결과 저장
    df_poisson.to_csv(output_file, index=False)
    print(f"  - 저장 파일: {output_file}")
    
    return df_poisson


def plot_dual_rate(input_file, output_file="dual_rate.png", 
                   bin_interval_ms=1000, moving_avg_window_s=1,
                   plot_duration_minutes=20.0):
    """
    요청 비율과 입력 토큰 비율을 함께 시각화합니다.
    (기존 코드와 동일)
    """
    # 파일 로드
    print(f"파일 '{input_file}' 로드 중...")
    df = pd.read_csv(input_file)
    
    # 타임스탬프 변환
    print("타임스탬프 변환 중...")
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='mixed', utc=True)
    
    # 필요한 컬럼 확인
    if 'ContextTokens' not in df.columns:
        print("경고: 'ContextTokens' 컬럼이 없습니다!")
        return
    
    # 결측값 처리
    df['ContextTokens'] = df['ContextTokens'].fillna(0)
    
    # 시간 범위 필터링
    if plot_duration_minutes is not None:
        start_time = df['TIMESTAMP'].min()
        end_time = start_time + pd.Timedelta(minutes=plot_duration_minutes)
        df_filtered = df[(df['TIMESTAMP'] >= start_time) & (df['TIMESTAMP'] <= end_time)]
        
        print(f"시간 범위 필터링: {plot_duration_minutes}분")
        print(f"  시작: {start_time}")
        print(f"  종료: {end_time}")
        print(f"  필터링 전 요청 수: {len(df)}")
        print(f"  필터링 후 요청 수: {len(df_filtered)}")
        
        df = df_filtered
    
    # 데이터 확인
    print(f"총 요청 수: {len(df)}")
    print(f"총 입력 토큰 수: {df['ContextTokens'].sum()}")
    print(f"평균 입력 토큰/요청: {df['ContextTokens'].mean():.2f}")
    print(f"기간: {df['TIMESTAMP'].min()} - {df['TIMESTAMP'].max()}")
    print(f"빈 간격: {bin_interval_ms}ms")
    
    # 상대 시간 계산 (초 단위)
    first_timestamp = df['TIMESTAMP'].min()
    df['relative_time'] = (df['TIMESTAMP'] - first_timestamp).dt.total_seconds()
    
    # 빈 간격 설정
    bin_interval_s = bin_interval_ms / 1000.0  # ms → s
    max_time = df['relative_time'].max()
    bins = np.arange(0, max_time + bin_interval_s, bin_interval_s)
    bin_centers = bins[:-1] + bin_interval_s / 2
    
    print(f"총 빈 개수: {len(bins)-1}")
    print(f"빈 간격: {bin_interval_s}초")
    
    # 요청 비율 계산
    relative_times = df['relative_time'].values
    request_counts, _ = np.histogram(relative_times, bins=bins)
    request_rate = request_counts / bin_interval_s
    
    # 입력 토큰 비율 계산
    token_sums = np.zeros(len(bins)-1)
    for i in range(len(bins)-1):
        bin_start = bins[i]
        bin_end = bins[i+1]
        mask = (df['relative_time'] >= bin_start) & (df['relative_time'] < bin_end)
        token_sums[i] = df[mask]['ContextTokens'].sum()
    
    token_rate = token_sums / bin_interval_s
    
    # 이동 평균 윈도우 계산
    moving_avg_window_bins = int(moving_avg_window_s / bin_interval_s)
    if moving_avg_window_bins < 1:
        moving_avg_window_bins = 1
    
    print(f"이동 평균 윈도우: {moving_avg_window_bins} 빈 ({moving_avg_window_s}초)")
    
    # 이동 평균 적용
    if moving_avg_window_bins > 1:
        request_rate_smoothed = np.convolve(request_rate, 
                                          np.ones(moving_avg_window_bins)/moving_avg_window_bins, 
                                          mode='same')
        token_rate_smoothed = np.convolve(token_rate, 
                                        np.ones(moving_avg_window_bins)/moving_avg_window_bins, 
                                        mode='same')
    else:
        request_rate_smoothed = request_rate
        token_rate_smoothed = token_rate
    
    # 통계 계산
    avg_request_rate = np.mean(request_rate)
    std_request_rate = np.std(request_rate)
    avg_token_rate = np.mean(token_rate)
    std_token_rate = np.std(token_rate)
    
    # 실제 타임스탬프로 변환 (그래프 x축용)
    timestamps = [first_timestamp + timedelta(seconds=t) for t in bin_centers]
    
    # 그래프 설정
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # 폰트 크기 설정
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14
    
    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=BIGGER_SIZE)
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=SMALL_SIZE)
    plt.rc('ytick', labelsize=SMALL_SIZE)
    plt.rc('legend', fontsize=SMALL_SIZE)
    
    # 첫 번째 서브플롯: 요청 비율
    ax1.plot(timestamps, request_rate_smoothed, '-', color='blue', linewidth=1.5, label='Request Rate')
    ax1.axhline(y=avg_request_rate, color='blue', linestyle='--', alpha=0.7, 
                label=f'Avg: {avg_request_rate:.2f} reqs/s\nStd: {std_request_rate:.2f}')
    
    ax1.set_ylabel('Request Rate (reqs/s)', fontsize=MEDIUM_SIZE)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # 제목에 시간 범위 정보 추가
    duration_str = f" (First {plot_duration_minutes} min)" if plot_duration_minutes else " (Full duration)"
    trace_type = "Poisson" if "poisson_" in os.path.basename(input_file) else "Original"
    ax1.set_title(f'{trace_type} Trace: Request Rate and Input Token Rate Over Time{duration_str}\n'
                  f'File: {os.path.basename(input_file)}', fontsize=BIGGER_SIZE)
    
    # 두 번째 서브플롯: 입력 토큰 비율
    ax2.plot(timestamps, token_rate_smoothed, '-', color='red', linewidth=1.5, label='Input Token Rate')
    ax2.axhline(y=avg_token_rate, color='red', linestyle='--', alpha=0.7, 
                label=f'Avg: {avg_token_rate:.0f} tokens/s\nStd: {std_token_rate:.0f}')
    
    ax2.set_ylabel('Input Token Rate (tokens/s)', fontsize=MEDIUM_SIZE)
    ax2.set_xlabel('Time', fontsize=MEDIUM_SIZE)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    # x축 형식 설정
    duration_seconds = max_time
    if plot_duration_minutes and plot_duration_minutes <= 60:
        ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))
        ax2.xaxis.set_major_locator(plt.matplotlib.dates.MinuteLocator(interval=5))
    elif plot_duration_minutes and plot_duration_minutes <= 180:
        ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        ax2.xaxis.set_major_locator(plt.matplotlib.dates.MinuteLocator(interval=15))
    else:
        ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        ax2.xaxis.set_major_locator(plt.matplotlib.dates.MinuteLocator(interval=30))
    
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    
    # 파일 저장
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"그래프가 '{output_file}'에 저장되었습니다.")
    
    # 통계 정보 출력
    print(f"\n=== 통계 정보 ===")
    print(f"총 요청 수: {len(df)}")
    print(f"총 입력 토큰 수: {df['ContextTokens'].sum():,}")
    print(f"빈 간격: {bin_interval_ms}ms")
    print(f"평균 요청 비율: {avg_request_rate:.2f} requests/second")
    print(f"요청 비율 표준편차: {std_request_rate:.2f}")
    print(f"최대 요청 비율: {np.max(request_rate_smoothed):.2f} requests/second")
    print(f"평균 토큰 비율: {avg_token_rate:.0f} tokens/second")
    print(f"토큰 비율 표준편차: {std_token_rate:.0f}")
    print(f"최대 토큰 비율: {np.max(token_rate_smoothed):.0f} tokens/second")
    print(f"평균 토큰/요청: {df['ContextTokens'].mean():.2f}")
    if plot_duration_minutes:
        print(f"플롯 시간 범위: {plot_duration_minutes}분")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Poisson process 기반 trace 생성 및 시각화')
    parser.add_argument('--input_file', default='azure_conv_0514_1400_20min_15.0x_tc.csv', type=str, help='입력 trace CSV 파일 경로')
    parser.add_argument('--output', type=str, help='출력 CSV 파일 경로 (기본값: poisson_{입력파일명})')
    parser.add_argument('--plot-output', type=str, help='출력 그래프 파일 경로')
    parser.add_argument('--bin-interval', type=int, default=1000, 
                        help='빈 간격(밀리초). 예: 10, 100, 1000')
    parser.add_argument('--avg-window', type=float, default=1.0, 
                        help='이동 평균 창 크기(초)')
    parser.add_argument('--plot-duration', type=int, default=20, 
                        help='플롯할 시간 범위(분). None이면 전체 데이터 사용')
    parser.add_argument('--seed', type=int, help='랜덤 시드 (재현 가능한 결과를 위해)')
    parser.add_argument('--generate-only', action='store_true', 
                        help='CSV 파일만 생성하고 그래프는 생성하지 않음')
    parser.add_argument('--plot-both', action='store_true', 
                        help='원본과 Poisson trace 모두 그래프 생성')
    
    args = parser.parse_args()
    
    # 입력 파일 확인
    if not os.path.exists(args.input_file):
        print(f"오류: 입력 파일 '{args.input_file}'이 존재하지 않습니다!")
        return
    
    # 출력 파일명 설정
    if args.output:
        output_csv = args.output
    else:
        input_basename = os.path.splitext(os.path.basename(args.input_file))[0]
        output_csv = f"poisson_{input_basename}.csv"
    
    # 랜덤 시드 설정
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"랜덤 시드 설정: {args.seed}")
    
    print(f"{'='*60}")
    print(f"Poisson Trace Generator")
    print(f"{'='*60}")
    
    # Poisson trace 생성
    print(f"입력: {args.input_file}")
    print(f"출력: {output_csv}")
    
    df_poisson = generate_poisson_trace(args.input_file, output_csv)
    
    if not args.generate_only:
        # 그래프 생성
        if args.plot_output:
            plot_output = args.plot_output
        else:
            plot_basename = os.path.splitext(output_csv)[0]
            plot_output = f"{plot_basename}.png"
        
        print(f"\n{'='*50}")
        print(f"그래프 생성")
        print(f"{'='*50}")
        
        # Poisson trace 그래프 생성
        plot_dual_rate(output_csv, plot_output, 
                      args.bin_interval, args.avg_window, args.plot_duration)
        
        # 원본과 비교 그래프도 생성
        if args.plot_both:
            print(f"\n--- 원본 trace 그래프 생성 ---")
            original_plot = f"original_{os.path.splitext(os.path.basename(args.input_file))[0]}.png"
            plot_dual_rate(args.input_file, original_plot, 
                          args.bin_interval, args.avg_window, args.plot_duration)
    
    print(f"\n{'='*60}")
    print(f"완료!")
    print(f"생성된 파일:")
    print(f"  - CSV: {output_csv}")
    if not args.generate_only:
        print(f"  - 그래프: {plot_output}")
        if args.plot_both:
            print(f"  - 원본 그래프: {original_plot}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()