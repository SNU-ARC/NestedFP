import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from datetime import datetime, timedelta
import os

def plot_request_rate(input_file, output_file="request_rate.pdf", view_mode="day", 
                      bin_interval_ms=1000, moving_avg_window_s=10, 
                      start_date=None, end_date=None):
    """
    트레이스 파일에서 요청 비율 그래프를 생성합니다.
    
    Args:
        input_file: 입력 CSV 파일 경로
        output_file: 출력 이미지 파일 경로
        view_mode: 'week', 'day', 'hour' 중 하나
        bin_interval_ms: 빈 간격(밀리초)
        moving_avg_window_s: 이동 평균 창 크기(초)
        start_date: 시작 날짜(YYYY-MM-DD 형식)
        end_date: 종료 날짜(YYYY-MM-DD 형식)
    """
    # 파일 로드
    print(f"파일 '{input_file}' 로드 중...")
    df = pd.read_csv(input_file)
    
    # 타임스탬프 변환
    print("타임스탬프 변환 중...")
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='mixed', utc=True)
    
    # 날짜 범위 필터링
    if start_date:
        start_datetime = pd.Timestamp(start_date, tz='UTC')
        df = df[df['TIMESTAMP'] >= start_datetime]
        print(f"시작 날짜 {start_date}로 필터링됨")
    
    if end_date:
        end_datetime = pd.Timestamp(end_date, tz='UTC') + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        df = df[df['TIMESTAMP'] <= end_datetime]
        print(f"종료 날짜 {end_date}로 필터링됨")
    
    # 데이터 확인
    if len(df) == 0:
        print("경고: 필터링 후 데이터가 없습니다!")
        return
    
    print(f"처리할 요청 수: {len(df)}")
    print(f"기간: {df['TIMESTAMP'].min()} - {df['TIMESTAMP'].max()}")
    
    # 시간 스케일에 따라 데이터 슬라이싱
    if view_mode == 'hour' and start_date is None:
        # 첫 시간만 사용
        start_hour = df['TIMESTAMP'].min().floor('H')
        end_hour = start_hour + pd.Timedelta(hours=1)
        df = df[(df['TIMESTAMP'] >= start_hour) & (df['TIMESTAMP'] < end_hour)]
        print(f"첫 1시간 데이터만 사용: {start_hour} - {end_hour}")
    elif view_mode == 'day':
        # day 모드일 때 정확히 하루(24시간)만 표시하기
        # 시작일의 자정(00:00)부터 다음날 자정까지
        if start_date:
            day_start = pd.Timestamp(start_date, tz='UTC')
        else:
            # 시작 날짜가 지정되지 않았다면 데이터의 첫 날짜 사용
            day_start = df['TIMESTAMP'].min().floor('D')  # 00:00으로 내림
        
        day_end = day_start + pd.Timedelta(days=1)  # 다음날 00:00
        
        # 정확히 하루만 필터링
        df = df[(df['TIMESTAMP'] >= day_start) & (df['TIMESTAMP'] < day_end)]
        print(f"정확히 하루(24시간) 데이터만 사용: {day_start} - {day_end}")
        
        # 데이터가 없는 경우 확인
        if len(df) == 0:
            print("경고: 선택한 날짜에 데이터가 없습니다!")
            return
    
    # 상대 시간 계산 (초 단위)
    first_timestamp = df['TIMESTAMP'].min()
    df['relative_time'] = (df['TIMESTAMP'] - first_timestamp).dt.total_seconds()
    
    # 빈 간격 설정
    bin_interval_s = bin_interval_ms / 1000.0  # ms → s
    max_time = df['relative_time'].max()
    bins = np.arange(0, max_time + bin_interval_s, bin_interval_s)
    
    # 히스토그램 빈닝
    relative_times = df['relative_time'].values
    request_counts, _ = np.histogram(relative_times, bins=bins)
    bin_centers = bins[:-1] + bin_interval_s / 2
    
    # 초당 요청 비율로 변환
    request_rate = request_counts / bin_interval_s
    
    # 이동 평균 윈도우 (초 단위 → 몇 개 빈을 평균?)
    moving_avg_window_bins = int(moving_avg_window_s / bin_interval_s)
    if moving_avg_window_bins < 1:
        moving_avg_window_bins = 1
    
    # 이동 평균 계산
    if moving_avg_window_bins > 1:
        request_rate_smoothed = np.convolve(request_rate, np.ones(moving_avg_window_bins)/moving_avg_window_bins, mode='same')
    else:
        request_rate_smoothed = request_rate
    
    # 실제 타임스탬프로 변환 (그래프 x축용)
    timestamps = [first_timestamp + timedelta(seconds=t) for t in bin_centers]
    
    # 그래프 설정
    fig_width = 8
    fig_height = 4
    
    plt.figure(figsize=(fig_width, fig_height))
    
    # 폰트 크기 설정
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16
    
    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=BIGGER_SIZE)
    plt.rc('axes', labelsize=BIGGER_SIZE)
    plt.rc('xtick', labelsize=MEDIUM_SIZE)
    plt.rc('ytick', labelsize=MEDIUM_SIZE)
    plt.rc('legend', fontsize=MEDIUM_SIZE)
    
    # 요청 비율 그래프 그리기
    plt.plot(timestamps, request_rate_smoothed, '-', color='blue', linewidth=1.5)
    
    # x축 형식 설정
    if view_mode == 'week':
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))
        plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.DayLocator())
        time_label = f"{first_timestamp.strftime('%Y-%m-%d')} to {(first_timestamp + timedelta(seconds=max_time)).strftime('%Y-%m-%d')}"
    elif view_mode == 'day':
        # 하루(24시간) 기준이므로 x축 범위 직접 설정
        if start_date:
            x_min = pd.Timestamp(start_date, tz='UTC')
        else:
            x_min = first_timestamp.floor('D')
        
        x_max = x_min + pd.Timedelta(days=1)
        plt.xlim(x_min, x_max)
        
        # 6시간 간격으로 레이블 표시 (00:00, 06:00, 12:00, 18:00, 24:00)
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.HourLocator(interval=3))
        
        # 보조 눈금 추가 (2시간 간격)
        plt.gca().xaxis.set_minor_locator(plt.matplotlib.dates.HourLocator(interval=3))
        plt.grid(True, which='major', linestyle='-', alpha=0.7)
        plt.grid(True, which='minor', linestyle=':', alpha=0.4)
        
        time_label = first_timestamp.strftime('%Y-%m-%d')
    elif view_mode == 'hour':
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MinuteLocator(interval=10))
        time_label = f"{first_timestamp.strftime('%Y-%m-%d %H:%M')} - {(first_timestamp + timedelta(seconds=max_time)).strftime('%H:%M')}"
    
    # x축 레이블이 겹치지 않도록 회전
    # plt.gcf().autofmt_xdate(rotation=45)
    
    # 그래프 레이블
    plt.xlabel('Time', fontsize=BIGGER_SIZE)
    plt.ylabel('Request Rate (reqs/s)', fontsize=BIGGER_SIZE)
    
    # 평균 요청 비율 계산 및 표시
    avg_rate = np.mean(request_rate)
    plt.axhline(y=avg_rate, color='r', linestyle='--', alpha=0.7, label=f'Average Request Rate')
    
    # 그래프 제목
    if view_mode == 'day':
        # plt.title(f"Request Rate Over 24 Hours ({time_label})\nBin: {bin_interval_ms}ms, Moving Avg: {moving_avg_window_s}s",fontsize=BIGGER_SIZE)
        pass
    else:
        plt.title(f"Request Rate Over Time ({time_label})\nBin: {bin_interval_ms}ms, Moving Avg: {moving_avg_window_s}s",
                fontsize=BIGGER_SIZE)
    
    # 범례
    plt.legend(loc='upper left')
    
    # 여백 조정
    plt.tight_layout()
    
    # save as png
    plt.savefig(output_file, dpi=300)
    
    print(f"그래프가 '{output_file}'에 저장되었습니다.")
    
    # 통계 정보 출력
    print("\n요청 비율 통계:")
    print(f"평균 요청 비율: {avg_rate:.2f} requests/second")
    print(f"최대 요청 비율: {np.max(request_rate_smoothed):.2f} requests/second")
    print(f"총 요청 수: {len(df)}")
    
    # 시간 간격별 분석 (사용자 요청 비율이 0이 아닌 간격의 비율)
    non_zero_intervals = np.sum(request_rate > 0)
    total_intervals = len(request_rate)
    print(f"활성 시간 비율: {non_zero_intervals / total_intervals:.2%} ({non_zero_intervals}/{total_intervals} 간격)")
    
    return plt

def main():
    parser = argparse.ArgumentParser(description='시간에 따른 요청 비율 시각화 도구')
    parser.add_argument('--file', type=str, 
                        default='/disk/dualfp_vllm_test/end_to_end_test/online/trace/AzureLLMInferenceTrace_code_1week.csv',
                        help='트레이스 파일 경로')
    parser.add_argument('--output', type=str, default='request_rate.pdf',
                        help='출력 파일 이름')
    parser.add_argument('--view', type=str, choices=['week', 'day', 'hour'], default='day',
                        help='시각화 스케일 (week/day/hour)')
    parser.add_argument('--bin', type=int, default=1000,
                        help='빈 간격(밀리초). 기본값: 1000ms = 1초')
    parser.add_argument('--avg-window', type=float, default=10.0,
                        help='이동 평균 창 크기(초). 기본값: 10초')
    parser.add_argument('--start-date', type=str, default=None,
                        help='시작 날짜 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='종료 날짜 (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    plot_request_rate(
        input_file=args.file,
        output_file=args.output,
        view_mode=args.view,
        bin_interval_ms=args.bin,
        moving_avg_window_s=args.avg_window,
        start_date=args.start_date,
        end_date=args.end_date
    )

if __name__ == "__main__":
    main()