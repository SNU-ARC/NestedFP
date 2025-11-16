import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from datetime import datetime, timedelta
import os


def scale_by_time_compression(input_file, output_file, scale_factor):
    """
    시간 간격을 일정 비율로 압축/확장하여 스케일링
    포아송 프로세스의 exponential 분포 성질을 유지
    
    Args:
        input_file: 입력 CSV 파일
        output_file: 출력 CSV 파일
        scale_factor: 시간 압축 비율 (0.2 = 5배 빠르게, 2.0 = 2배 느리게)
    """
    print(f"\n[Time Compression] 시작: scale_factor={scale_factor}")
    
    # 파일 로드
    df = pd.read_csv(input_file)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='mixed', utc=True)
    
    # 통계 정보
    original_count = len(df)
    original_duration = (df['TIMESTAMP'].max() - df['TIMESTAMP'].min()).total_seconds()
    original_rate = original_count / original_duration
    
    print(f"원본 요청 수: {original_count}")
    print(f"원본 기간: {original_duration:.2f}초")
    print(f"원본 평균 요청 비율: {original_rate:.2f} reqs/s")
    
    # 시간 압축
    df_scaled = df.copy()
    first_time = df['TIMESTAMP'].iloc[0]
    
    # 상대 시간 계산 후 스케일링
    relative_times = (df['TIMESTAMP'] - first_time).dt.total_seconds()
    scaled_times = relative_times * scale_factor
    
    # 새로운 타임스탬프 생성
    df_scaled['TIMESTAMP'] = first_time + pd.to_timedelta(scaled_times, unit='s')
    
    # 결과 저장
    df_scaled.to_csv(output_file, index=False)
    
    # 결과 통계
    new_duration = (df_scaled['TIMESTAMP'].max() - df_scaled['TIMESTAMP'].min()).total_seconds()
    new_rate = original_count / new_duration
    
    print(f"[Time Compression] 결과:")
    print(f"  - 새 기간: {new_duration:.2f}초")
    print(f"  - 새 요청 비율: {new_rate:.2f} reqs/s")
    print(f"  - 비율 변화: {new_rate/original_rate:.2f}x")
    print(f"  - 저장 파일: {output_file}")
    
    return df_scaled

def scale_by_round_robin(input_file, output_file, scale_factor):
    """
    Round-robin 방식으로 요청을 균등하게 샘플링
    
    Args:
        input_file: 입력 CSV 파일
        output_file: 출력 CSV 파일
        scale_factor: 유지할 비율 (0.2 = 20%만 유지)
    """
    print(f"\n[Round Robin] 시작: scale_factor={scale_factor}")
    
    # 파일 로드
    df = pd.read_csv(input_file)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='mixed', utc=True)
    
    # 통계 정보
    original_count = len(df)
    original_duration = (df['TIMESTAMP'].max() - df['TIMESTAMP'].min()).total_seconds()
    original_rate = original_count / original_duration
    
    print(f"원본 요청 수: {original_count}")
    print(f"원본 기간: {original_duration:.2f}초")
    print(f"원본 평균 요청 비율: {original_rate:.2f} reqs/s")
    
    # Round-robin 샘플링
    step = max(1, int(1 / scale_factor))
    indices = np.arange(0, original_count, step)
    df_scaled = df.iloc[indices].copy()
    df_scaled.reset_index(drop=True, inplace=True)
    
    # 결과 저장
    df_scaled.to_csv(output_file, index=False)
    
    # 결과 통계
    new_count = len(df_scaled)
    new_rate = new_count / original_duration  # 기간은 동일하게 유지
    
    print(f"[Round Robin] 결과:")
    print(f"  - Step: {step} (매 {step}번째 요청 선택)")
    print(f"  - 새 요청 수: {new_count} / {original_count} ({new_count/original_count*100:.1f}%)")
    print(f"  - 새 요청 비율: {new_rate:.2f} reqs/s")
    print(f"  - 비율 변화: {new_rate/original_rate:.2f}x")
    print(f"  - 저장 파일: {output_file}")
    
    return df_scaled

def scale_by_token_length(input_file, output_file, scale_factor):
    """
    토큰 길이를 일정 비율로 확장하여 스케일링
    ContextTokens, GeneratedTokens 등 토큰 관련 컬럼들을 확장
    
    Args:
        input_file: 입력 CSV 파일
        output_file: 출력 CSV 파일
        scale_factor: 토큰 확장 비율 (4.0 = 4배 확장, 0.5 = 절반으로 축소)
    """
    print(f"\n[Token Length Scaling] 시작: scale_factor={scale_factor}")
    
    # 파일 로드
    df = pd.read_csv(input_file)
    if 'TIMESTAMP' in df.columns:
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='mixed', utc=True)
    
    # 토큰 관련 컬럼 찾기
    token_columns = []
    for col in df.columns:
        if 'token' in col.lower() or 'Token' in col:
            token_columns.append(col)
    
    if not token_columns:
        print("경고: 토큰 관련 컬럼을 찾을 수 없습니다!")
        # 일반적인 토큰 컬럼명들 시도
        possible_token_cols = ['ContextTokens', 'GeneratedTokens', 'InputTokens', 'OutputTokens', 'TotalTokens']
        for col in possible_token_cols:
            if col in df.columns:
                token_columns.append(col)
    
    print(f"발견된 토큰 컬럼들: {token_columns}")
    
    if not token_columns:
        print("오류: 토큰 관련 컬럼을 찾을 수 없어 처리할 수 없습니다!")
        return None
    
    # 통계 정보
    df_scaled = df.copy()
    
    print(f"원본 요청 수: {len(df)}")
    for col in token_columns:
        original_sum = df[col].sum() if col in df.columns else 0
        original_avg = df[col].mean() if col in df.columns else 0
        print(f"  - {col}: 총합={original_sum:,}, 평균={original_avg:.2f}")
    
    # 토큰 길이 스케일링
    for col in token_columns:
        if col in df_scaled.columns:
            # 결측값 처리
            df_scaled[col] = df_scaled[col].fillna(0)
            # 스케일링 (정수로 반올림)
            df_scaled[col] = (df_scaled[col] * scale_factor).round().astype(int)
    
    # 결과 저장
    df_scaled.to_csv(output_file, index=False)
    
    # 결과 통계
    print(f"[Token Length Scaling] 결과:")
    for col in token_columns:
        if col in df_scaled.columns:
            new_sum = df_scaled[col].sum()
            new_avg = df_scaled[col].mean()
            original_sum = df[col].sum() if col in df.columns else 0
            original_avg = df[col].mean() if col in df.columns else 0
            print(f"  - {col}: 총합={new_sum:,} (변화: {new_sum/original_sum:.2f}x), 평균={new_avg:.2f} (변화: {new_avg/original_avg:.2f}x)")
    print(f"  - 저장 파일: {output_file}")
    
    return df_scaled

def create_scaled_files(base_filename, time_scale_factors=None, token_scale_factors=None, 
                       scaling_method="time_compression"):
    """
    스케일된 CSV 파일들을 생성합니다.
    
    Args:
        base_filename: 기본 파일명 (확장자 제외)
        time_scale_factors: 시간 스케일 팩터 리스트
        token_scale_factors: 토큰 스케일 팩터 리스트
        scaling_method: "time_compression" 또는 "round_robin" (시간 스케일링용)
    """
    input_file = f"{base_filename}.csv"
    
    if not os.path.exists(input_file):
        print(f"오류: {input_file}이 존재하지 않습니다!")
        return
    
    print(f"\n{'='*60}")
    print(f"스케일된 파일 생성 시작")
    print(f"기본 파일: {input_file}")
    print(f"시간 스케일링 방법: {scaling_method}")
    print(f"시간 스케일 팩터: {time_scale_factors}")
    print(f"토큰 스케일 팩터: {token_scale_factors}")
    print(f"{'='*60}")
    
    # 파일명 접미사 결정
    if scaling_method == "time_compression":
        time_suffix = "tc"
    else:
        time_suffix = "rr"
    
    # 시간 스케일링만 수행
    if time_scale_factors and not token_scale_factors:
        for time_scale in time_scale_factors:
            output_file = f"{base_filename}_{time_scale}x_{time_suffix}.csv"
            print(f"\n--- 시간 스케일링: time_scale={time_scale} ---")
            
            if scaling_method == "time_compression":
                scale_by_time_compression(input_file, output_file, time_scale)
            elif scaling_method == "round_robin":
                scale_by_round_robin(input_file, output_file, time_scale)
    
    # 토큰 스케일링만 수행
    elif token_scale_factors and not time_scale_factors:
        for token_scale in token_scale_factors:
            output_file = f"{base_filename}_{token_scale}x_token.csv"
            print(f"\n--- 토큰 스케일링: token_scale={token_scale} ---")
            scale_by_token_length(input_file, output_file, token_scale)
    
    # 시간과 토큰 스케일링 모두 수행
    elif time_scale_factors and token_scale_factors:
        for time_scale in time_scale_factors:
            for token_scale in token_scale_factors:
                # 먼저 시간 스케일링
                temp_file = f"{base_filename}_{time_scale}x_{time_suffix}_temp.csv"
                if scaling_method == "time_compression":
                    scale_by_time_compression(input_file, temp_file, time_scale)
                elif scaling_method == "round_robin":
                    scale_by_round_robin(input_file, temp_file, time_scale)
                
                # 그 다음 토큰 스케일링
                output_file = f"{base_filename}_{time_scale}x_{time_suffix}_{token_scale}x_token.csv"
                print(f"\n--- 복합 스케일링: time_scale={time_scale}, token_scale={token_scale} ---")
                scale_by_token_length(temp_file, output_file, token_scale)
                
                # 임시 파일 삭제
                os.remove(temp_file)
    
    else:
        print("오류: time_scale_factors 또는 token_scale_factors 중 최소 하나는 제공되어야 합니다!")


def plot_dual_rate(input_file, output_file="dual_rate.png", 
                   bin_interval_ms=1000, moving_avg_window_s=1,
                   plot_duration_minutes=20.0):
    """
    요청 비율과 입력 토큰 비율을 함께 시각화합니다.
    
    Args:
        input_file: 입력 CSV 파일 경로
        output_file: 출력 이미지 파일 경로
        bin_interval_ms: 빈 간격(밀리초) - 10ms, 100ms, 1000ms 등
        moving_avg_window_s: 이동 평균 창 크기(초)
        plot_duration_minutes: 플롯할 시간 범위(분), None이면 전체 데이터 사용
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
    # 각 빈에 대해 해당 시간대의 토큰 합계 계산
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
    var_request_rate = np.var(request_rate)
    std_request_rate = np.std(request_rate)
    
    avg_token_rate = np.mean(token_rate)
    var_token_rate = np.var(token_rate)
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
    ax1.set_title(f'Request Rate and Input Token Rate Over Time{duration_str}\n'
                  f'File: {os.path.basename(input_file)}', fontsize=BIGGER_SIZE)
    
    # 두 번째 서브플롯: 입력 토큰 비율
    ax2.plot(timestamps, token_rate_smoothed, '-', color='red', linewidth=1.5, label='Input Token Rate')
    ax2.axhline(y=avg_token_rate, color='red', linestyle='--', alpha=0.7, 
                label=f'Avg: {avg_token_rate:.0f} tokens/s\nStd: {std_token_rate:.0f}')
    
    ax2.set_ylabel('Input Token Rate (tokens/s)', fontsize=MEDIUM_SIZE)
    ax2.set_xlabel('Time', fontsize=MEDIUM_SIZE)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    # x축 형식 설정 - 시간 범위에 따라 조정
    duration_seconds = max_time
    if plot_duration_minutes and plot_duration_minutes <= 60:  # 1시간 이하
        ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))
        ax2.xaxis.set_major_locator(plt.matplotlib.dates.MinuteLocator(interval=5))
    elif plot_duration_minutes and plot_duration_minutes <= 180:  # 3시간 이하
        ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        ax2.xaxis.set_major_locator(plt.matplotlib.dates.MinuteLocator(interval=15))
    else:  # 긴 시간
        ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        ax2.xaxis.set_major_locator(plt.matplotlib.dates.MinuteLocator(interval=30))
    
    # x축 레이블 회전
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 여백 조정
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
    print(f"요청 비율 분산: {var_request_rate:.2f}")
    print(f"요청 비율 표준편차: {std_request_rate:.2f}")
    print(f"최대 요청 비율: {np.max(request_rate_smoothed):.2f} requests/second")
    print(f"평균 토큰 비율: {avg_token_rate:.0f} tokens/second")
    print(f"토큰 비율 분산: {var_token_rate:.0f}")
    print(f"토큰 비율 표준편차: {std_token_rate:.0f}")
    print(f"최대 토큰 비율: {np.max(token_rate_smoothed):.0f} tokens/second")
    print(f"평균 토큰/요청: {df['ContextTokens'].mean():.2f}")
    if plot_duration_minutes:
        print(f"플롯 시간 범위: {plot_duration_minutes}분")
    
    plt.close()


def plot_all_scaled_traces(base_filename="azure_conv_0514_1400_20min", 
                          time_scale_factors=None, token_scale_factors=None,
                          bin_interval_ms=1000, moving_avg_window_s=10,
                          scaling_method="time_compression", plot_duration_minutes=20):
    """
    모든 스케일된 트레이스 파일들을 한 번에 플롯합니다.
    """
    files_to_plot = []
    
    # 원본 파일
    original_file = f"{base_filename}.csv"
    if os.path.exists(original_file):
        files_to_plot.append((original_file, f"{base_filename}.png"))
    
    # 파일명 접미사 결정
    if scaling_method == "time_compression":
        time_suffix = "tc"
    else:
        time_suffix = "rr"
    
    # 시간 스케일링만 수행된 파일들
    if time_scale_factors and not token_scale_factors:
        for time_scale in time_scale_factors:
            input_file = f"{base_filename}_{time_scale}x_{time_suffix}.csv"
            output_file = f"{base_filename}_{time_scale}x_{time_suffix}.png"
            if os.path.exists(input_file):
                files_to_plot.append((input_file, output_file))
    
    # 토큰 스케일링만 수행된 파일들
    elif token_scale_factors and not time_scale_factors:
        for token_scale in token_scale_factors:
            input_file = f"{base_filename}_{token_scale}x_token.csv"
            output_file = f"{base_filename}_{token_scale}x_token.png"
            if os.path.exists(input_file):
                files_to_plot.append((input_file, output_file))
    
    # 시간과 토큰 스케일링 모두 수행된 파일들
    elif time_scale_factors and token_scale_factors:
        for time_scale in time_scale_factors:
            for token_scale in token_scale_factors:
                input_file = f"{base_filename}_{time_scale}x_{time_suffix}_{token_scale}x_token.csv"
                output_file = f"{base_filename}_{time_scale}x_{time_suffix}_{token_scale}x_token.png"
                if os.path.exists(input_file):
                    files_to_plot.append((input_file, output_file))
    
    # 파일들 플롯
    for input_file, output_file in files_to_plot:
        print(f"\n{'='*50}")
        print(f"Processing: {input_file}")
        print(f"{'='*50}")
        plot_dual_rate(input_file, output_file, bin_interval_ms, moving_avg_window_s, plot_duration_minutes)


def main():
    parser = argparse.ArgumentParser(description='요청 비율과 입력 토큰 비율 동시 시각화 + 토큰 길이 스케일링')
    parser.add_argument('--file', type=str, help='단일 트레이스 파일 경로')
    parser.add_argument('--output', type=str, default='dual_rate.png', help='출력 파일 이름')
    
    # 더 직관적인 이름으로 변경
    parser.add_argument('--generate-and-plot', action='store_true', 
                        help='스케일된 파일들을 생성하고 모든 그래프를 생성 (기본 동작)')
    parser.add_argument('--generate-only', action='store_true', 
                        help='스케일된 CSV 파일들만 생성')
    parser.add_argument('--plot-only', action='store_true', 
                        help='기존 스케일된 파일들의 그래프만 생성')
    
    parser.add_argument('--base-name', type=str, default='azure_conv_0514_1400_20min',
                        help='기본 파일명 (확장자 제외)')
    parser.add_argument('--bin-interval', type=int, default=1000, 
                        help='빈 간격(밀리초). 예: 10, 100, 1000')
    parser.add_argument('--avg-window', type=float, default=1.0, help='이동 평균 창 크기(초)')
    parser.add_argument('--plot-duration', type=int, default=20, 
                        help='플롯할 시간 범위(분). None이면 전체 데이터 사용')
    parser.add_argument('--scaling-method', type=str, default='time_compression',
                        choices=['time_compression', 'round_robin'],
                        help='시간 스케일링 방법 선택')
    
    # 시간과 토큰 스케일링을 분리
    parser.add_argument('--time-scale-factors', type=float, nargs='+',
                        default=[15.0, 16.0, 17.0],
                        help='시간 스케일 팩터 리스트 (예: --time-scale-factors 0.2 0.4 0.8)')
    parser.add_argument('--token-scale-factors', type=float, nargs='+', 
                        default=[4.0, 8.0, 16.0],
                        help='토큰 스케일 팩터 리스트 (예: --token-scale-factors 4.0 8.0 16.0)')
    
    # 기존 호환성을 위한 옵션 (deprecated)
    parser.add_argument('--scale-factors', type=float, nargs='+', 
                        default=None,
                        help='(deprecated) 시간 스케일 팩터 리스트. --time-scale-factors 사용 권장')
    
    args = parser.parse_args()
    
    # 기존 호환성 처리
    if args.scale_factors and not args.time_scale_factors:
        print("경고: --scale-factors는 deprecated입니다. --time-scale-factors를 사용하세요.")
        args.time_scale_factors = args.scale_factors
    
    # 빈 간격 유효성 검사
    if args.bin_interval <= 0:
        print("오류: 빈 간격은 0보다 커야 합니다.")
        return
    
    if args.bin_interval < 10:
        print("경고: 매우 작은 빈 간격(<10ms)은 성능에 영향을 줄 수 있습니다.")
    
    print(f"빈 간격: {args.bin_interval}ms")
    print(f"이동 평균 창: {args.avg_window}초")
    print(f"시간 스케일링 방법: {args.scaling_method}")
    print(f"시간 스케일 팩터: {args.time_scale_factors}")
    print(f"토큰 스케일 팩터: {args.token_scale_factors}")
    
    # 기본 동작: 아무 옵션도 지정하지 않으면 generate-and-plot
    if not any([args.generate_only, args.plot_only, args.file]):
        args.generate_and_plot = True
    
    # 단일 파일 처리
    if args.file:
        print(f"\n=== 단일 파일 처리 모드 ===")
        plot_dual_rate(args.file, args.output, args.bin_interval, args.avg_window, args.plot_duration)
        return
        
    if args.generate_only:
        # CSV 파일들만 생성
        print("\n=== CSV 파일 생성 모드 ===")
        create_scaled_files(args.base_name, args.time_scale_factors, args.token_scale_factors, args.scaling_method)
        
    elif args.plot_only:
        # 그래프만 생성
        print("\n=== 그래프 생성 모드 ===")
        plot_all_scaled_traces(base_filename=args.base_name, 
                              time_scale_factors=args.time_scale_factors,
                              token_scale_factors=args.token_scale_factors,
                              bin_interval_ms=args.bin_interval,
                              moving_avg_window_s=args.avg_window,
                              scaling_method=args.scaling_method,
                              plot_duration_minutes=args.plot_duration)
        
    else:  # generate_and_plot (기본)
        print("\n=== CSV 생성 + 그래프 생성 모드 (기본) ===")
        
        # 1단계: 스케일된 파일들 생성
        print("\n--- 1단계: 스케일된 CSV 파일들 생성 ---")
        create_scaled_files(args.base_name, args.time_scale_factors, args.token_scale_factors, args.scaling_method)
        
        # 2단계: 모든 파일들의 그래프 생성
        print("\n--- 2단계: 모든 파일들의 그래프 생성 ---")
        plot_all_scaled_traces(base_filename=args.base_name, 
                              time_scale_factors=args.time_scale_factors,
                              token_scale_factors=args.token_scale_factors,
                              bin_interval_ms=args.bin_interval,
                              moving_avg_window_s=args.avg_window,
                              scaling_method=args.scaling_method,
                              plot_duration_minutes=args.plot_duration)
        
        print("\n=== 완료! ===")
        

if __name__ == "__main__":
    main()