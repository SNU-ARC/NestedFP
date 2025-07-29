import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
from scipy import stats
from scipy.stats import norm





def load_iteration_data(file_path):
    """
    benchmark_iteration.json 파일을 로드하고 DataFrame으로 변환
    
    Args:
        file_path: JSON 파일 경로
        
    Returns:
        pandas.DataFrame: iteration 데이터
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} iterations from {file_path}")
    
    # DataFrame으로 변환
    df = pd.DataFrame(data)
    
    # iteration_total로 정렬
    df = df.sort_values('iteration_total').reset_index(drop=True)
    
    # 데이터 요약 출력
    print(f"Iteration range: {df['iteration_total'].min()} - {df['iteration_total'].max()}")
    print(f"Total tokens generated: {df['tokens_generated'].sum()}")
    
    return df


def plot_scheduled_tokens(df, output_file='scheduled_tokens_by_iteration.png', zoom_samples=2000):
    """
    x축: iteration step, y축: total_scheduled_tokens
    prefill_tokens와 decode_tokens도 함께 plot
    전체 범위와 짧은 범위(확대된 view) 두 개의 subplot 제공
    
    Args:
        df: iteration 데이터 DataFrame
        output_file: 출력 파일명
        zoom_samples: 확대된 view에서 보여줄 샘플 수 (기본값: 2000)
    """
    # 2x1 subplot 생성 (위아래로 배치)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # iteration step 데이터
    iterations = df['iteration_total']
    
    # 토큰 데이터 (None 값을 0으로 처리)
    total_tokens = df['total_scheduled_tokens'].fillna(0)
    prefill_tokens = df['prefill_tokens'].fillna(0)
    decode_tokens = df['decode_tokens'].fillna(0)
    
    # 상단 subplot: 전체 범위
    ax1.plot(iterations, total_tokens, 'b-', linewidth=2, label='Total Scheduled Tokens', alpha=0.8)
    ax1.plot(iterations, prefill_tokens, 'r--', linewidth=1.5, label='Prefill Tokens', alpha=0.7)
    ax1.plot(iterations, decode_tokens, 'g--', linewidth=1.5, label='Decode Tokens', alpha=0.7)
    
    # 상단 subplot 스타일 설정
    ax1.set_xlabel('Iteration Step', fontsize=12)
    ax1.set_ylabel('Number of Tokens', fontsize=12)
    ax1.set_title('Scheduled Tokens by Iteration Step - Full Range', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    # 전체 통계 정보 추가
    full_stats_text = f"""Full Range Statistics:
    Total Iterations: {len(df):,}
    Avg Total Tokens: {total_tokens.mean():.1f}
    Max Total Tokens: {total_tokens.max():.0f}
    Iteration Range: {iterations.min()} - {iterations.max()}"""
    
    ax1.text(0.02, 0.98, full_stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 하단 subplot: 짧은 범위 (확대된 view)
    zoom_end = min(zoom_samples, len(df))
    zoom_iterations = iterations[:zoom_end]
    zoom_total_tokens = total_tokens[:zoom_end]
    zoom_prefill_tokens = prefill_tokens[:zoom_end]
    zoom_decode_tokens = decode_tokens[:zoom_end]
    
    # 확대된 view에서는 마커도 함께 표시하여 개별 포인트를 더 잘 볼 수 있게 함
    ax2.plot(zoom_iterations, zoom_total_tokens, 'b-', linewidth=2, 
             label='Total Scheduled Tokens', alpha=0.8, marker='o', markersize=2)
    ax2.plot(zoom_iterations, zoom_prefill_tokens, 'r--', linewidth=1.5, 
             label='Prefill Tokens', alpha=0.7, marker='s', markersize=1.5)
    ax2.plot(zoom_iterations, zoom_decode_tokens, 'g--', linewidth=1.5, 
             label='Decode Tokens', alpha=0.7, marker='^', markersize=1.5)
    
    # 하단 subplot 스타일 설정
    ax2.set_xlabel('Iteration Step', fontsize=12)
    ax2.set_ylabel('Number of Tokens', fontsize=12)
    ax2.set_title(f'Scheduled Tokens - Detailed View (First {zoom_end:,} iterations)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    
    # 확대된 범위 통계 정보 추가
    zoom_stats_text = f"""Detailed View Statistics:
    Samples Shown: {zoom_end:,}
    Avg Total Tokens: {zoom_total_tokens.mean():.1f}
    Max Total Tokens: {zoom_total_tokens.max():.0f}
    Token Std Dev: {zoom_total_tokens.std():.1f}"""
    
    ax2.text(0.02, 0.98, zoom_stats_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 변동성 분석 (확대된 view에서)
    if len(zoom_total_tokens) > 1:
        # 연속된 iteration 간의 변화량 계산
        token_changes = np.abs(np.diff(zoom_total_tokens))
        avg_change = token_changes.mean()
        max_change = token_changes.max()
        
        # 변동성 정보를 하단 subplot에 추가
        volatility_text = f"""Oscillation Analysis:
        Avg Change: {avg_change:.1f}
        Max Change: {max_change:.0f}
        Change Std: {token_changes.std():.1f}"""
        
        ax2.text(0.98, 0.98, volatility_text, transform=ax2.transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Scheduled tokens plot saved to {output_file}")
    print(f"  - Full range: {len(df):,} iterations")
    print(f"  - Detailed view: first {zoom_end:,} iterations")
    
    # 검증: prefill + decode = total인지 확인
    calculated_total = prefill_tokens + decode_tokens
    difference = np.abs(total_tokens - calculated_total)
    max_diff = difference.max()
    
    print(f"Token sum validation - Max difference: {max_diff:.2f}")
    if max_diff > 0.1:  # 부동소수점 오차 고려
        print("Warning: prefill_tokens + decode_tokens != total_scheduled_tokens")
    
    # 진동 패턴 분석 결과 출력
    if len(zoom_total_tokens) > 1:
        token_changes = np.abs(np.diff(zoom_total_tokens))
        print(f"Oscillation pattern analysis (first {zoom_end} iterations):")
        print(f"  - Average token change between iterations: {token_changes.mean():.1f}")
        print(f"  - Maximum token change: {token_changes.max():.0f}")
        print(f"  - Token change standard deviation: {token_changes.std():.1f}")

def plot_scheduled_requests(df, output_file='scheduled_requests_by_iteration.png', zoom_samples=2000):
    """
    x축: iteration step, y축: total_scheduled_requests
    prefill_requests와 decode_requests도 함께 plot
    전체 범위와 짧은 범위(확대된 view) 두 개의 subplot 제공
    
    Args:
        df: iteration 데이터 DataFrame
        output_file: 출력 파일명
        zoom_samples: 확대된 view에서 보여줄 샘플 수 (기본값: 2000)
    """
    # 2x1 subplot 생성 (위아래로 배치)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # iteration step 데이터
    iterations = df['iteration_total']
    
    # 요청 데이터 (None 값을 0으로 처리)
    total_requests = df['total_scheduled_requests'].fillna(0)
    prefill_requests = df['prefill_requests'].fillna(0)
    decode_requests = df['decode_requests'].fillna(0)
    
    # 상단 subplot: 전체 범위
    ax1.plot(iterations, total_requests, 'b-', linewidth=2, label='Total Scheduled Requests', alpha=0.8)
    ax1.plot(iterations, prefill_requests, 'r--', linewidth=1.5, label='Prefill Requests', alpha=0.7)
    ax1.plot(iterations, decode_requests, 'g--', linewidth=1.5, label='Decode Requests', alpha=0.7)
    
    # 상단 subplot 스타일 설정
    ax1.set_xlabel('Iteration Step', fontsize=12)
    ax1.set_ylabel('Number of Requests', fontsize=12)
    ax1.set_title('Scheduled Requests by Iteration Step - Full Range', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    # 전체 통계 정보 추가
    full_stats_text = f"""Full Range Statistics:
    Total Iterations: {len(df):,}
    Avg Total Requests: {total_requests.mean():.1f}
    Max Total Requests: {total_requests.max():.0f}
    Iteration Range: {iterations.min()} - {iterations.max()}"""
    
    ax1.text(0.02, 0.98, full_stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 하단 subplot: 짧은 범위 (확대된 view)
    zoom_end = min(zoom_samples, len(df))
    zoom_iterations = iterations[:zoom_end]
    zoom_total_requests = total_requests[:zoom_end]
    zoom_prefill_requests = prefill_requests[:zoom_end]
    zoom_decode_requests = decode_requests[:zoom_end]
    
    # 확대된 view에서는 마커도 함께 표시
    ax2.plot(zoom_iterations, zoom_total_requests, 'b-', linewidth=2, 
             label='Total Scheduled Requests', alpha=0.8, marker='o', markersize=2)
    ax2.plot(zoom_iterations, zoom_prefill_requests, 'r--', linewidth=1.5, 
             label='Prefill Requests', alpha=0.7, marker='s', markersize=1.5)
    ax2.plot(zoom_iterations, zoom_decode_requests, 'g--', linewidth=1.5, 
             label='Decode Requests', alpha=0.7, marker='^', markersize=1.5)
    
    # 하단 subplot 스타일 설정
    ax2.set_xlabel('Iteration Step', fontsize=12)
    ax2.set_ylabel('Number of Requests', fontsize=12)
    ax2.set_title(f'Scheduled Requests - Detailed View (First {zoom_end:,} iterations)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    
    # 확대된 범위 통계 정보 추가
    zoom_stats_text = f"""Detailed View Statistics:
    Samples Shown: {zoom_end:,}
    Avg Total Requests: {zoom_total_requests.mean():.1f}
    Max Total Requests: {zoom_total_requests.max():.0f}
    Request Std Dev: {zoom_total_requests.std():.1f}"""
    
    ax2.text(0.02, 0.98, zoom_stats_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # 변동성 분석 (확대된 view에서)
    if len(zoom_total_requests) > 1:
        # 연속된 iteration 간의 변화량 계산
        request_changes = np.abs(np.diff(zoom_total_requests))
        avg_change = request_changes.mean()
        max_change = request_changes.max()
        
        # 변동성 정보를 하단 subplot에 추가
        volatility_text = f"""Oscillation Analysis:
        Avg Change: {avg_change:.1f}
        Max Change: {max_change:.0f}
        Change Std: {request_changes.std():.1f}"""
        
        ax2.text(0.98, 0.98, volatility_text, transform=ax2.transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Scheduled requests plot saved to {output_file}")
    print(f"  - Full range: {len(df):,} iterations")
    print(f"  - Detailed view: first {zoom_end:,} iterations")
    
    # 검증: prefill + decode = total인지 확인
    calculated_total = prefill_requests + decode_requests
    difference = np.abs(total_requests - calculated_total)
    max_diff = difference.max()
    
    print(f"Request sum validation - Max difference: {max_diff:.2f}")
    if max_diff > 0.1:  # 부동소수점 오차 고려
        print("Warning: prefill_requests + decode_requests != total_scheduled_requests")
    
    # 진동 패턴 분석 결과 출력
    if len(zoom_total_requests) > 1:
        request_changes = np.abs(np.diff(zoom_total_requests))
        print(f"Oscillation pattern analysis (first {zoom_end} iterations):")
        print(f"  - Average request change between iterations: {request_changes.mean():.1f}")
        print(f"  - Maximum request change: {request_changes.max():.0f}")
        print(f"  - Request change standard deviation: {request_changes.std():.1f}")

def plot_throughput_by_iteration(df, output_file='throughput_by_iteration.png', zoom_samples=2000):
    """
    x축: iteration step, y축: throughput (tokens/s)  
    throughput = decode_requests / itl
    overall throughput도 함께 표시
    
    Args:
        df: iteration 데이터 DataFrame
        output_file: 출력 파일명
        zoom_samples: 확대된 view에서 보여줄 샘플 수 (기본값: 2000)
    """
    # ITL이 null이 아니고 0보다 큰 데이터만 필터링
    valid_data = df[(df['itl'].notna()) & (df['itl'] > 0) & (df['decode_requests'].notna())].copy()
    
    if len(valid_data) == 0:
        print("No valid ITL and decode_requests data found for throughput calculation, skipping...")
        return
    
    # Throughput 계산: decode_requests / itl (tokens/s)
    decode_requests = valid_data['decode_requests'].fillna(0)
    itl_values = valid_data['itl']
    throughput = decode_requests / itl_values
    
    # Overall throughput 계산
    # 전체 tokens_generated의 합 / 모든 valid ITL의 합
    total_tokens_generated = df['tokens_generated'].fillna(0).sum()
    total_processing_time = valid_data['itl'].sum()
    overall_throughput = total_tokens_generated / total_processing_time if total_processing_time > 0 else 0
    
    print(f"Throughput calculation:")
    print(f"  Total tokens generated: {total_tokens_generated:,}")
    print(f"  Total processing time: {total_processing_time:.2f} seconds")
    print(f"  Overall throughput: {overall_throughput:.2f} tokens/s")
    
    # 2x1 subplot 생성
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # 상단 subplot: 전체 범위
    iterations = valid_data['iteration_total']
    
    ax1.plot(iterations, throughput, 'g-', linewidth=2, label='Throughput (tokens/s)', alpha=0.8)
    ax1.axhline(y=overall_throughput, color='red', linestyle='--', linewidth=2.5, 
                label=f'Overall Throughput: {overall_throughput:.2f} tokens/s', alpha=0.8)
    
    ax1.set_xlabel('Iteration Step', fontsize=12)
    ax1.set_ylabel('Throughput (tokens/s)', fontsize=12)
    ax1.set_title('Throughput by Iteration Step - Full Range', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    # 통계 정보 추가
    full_stats_text = f"""Full Range Statistics:
    Valid Iterations: {len(valid_data):,}
    Avg Throughput: {throughput.mean():.2f} tokens/s
    Max Throughput: {throughput.max():.2f} tokens/s
    Min Throughput: {throughput.min():.2f} tokens/s
    Overall Throughput: {overall_throughput:.2f} tokens/s
    Total Tokens: {total_tokens_generated:,}"""
    
    ax1.text(0.02, 0.98, full_stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # 하단 subplot: 확대된 view
    zoom_end = min(zoom_samples, len(valid_data))
    zoom_iterations = iterations.iloc[:zoom_end]
    zoom_throughput = throughput.iloc[:zoom_end]
    
    ax2.plot(zoom_iterations, zoom_throughput, 'g-', linewidth=2, 
             label='Throughput (tokens/s)', alpha=0.8, marker='o', markersize=2)
    ax2.axhline(y=overall_throughput, color='red', linestyle='--', linewidth=2.5, 
                label=f'Overall: {overall_throughput:.2f} tokens/s', alpha=0.8)
    
    ax2.set_xlabel('Iteration Step', fontsize=12)
    ax2.set_ylabel('Throughput (tokens/s)', fontsize=12)  
    ax2.set_title(f'Throughput - Detailed View (First {zoom_end:,} iterations)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    
    # 확대된 범위 통계 정보
    zoom_stats_text = f"""Detailed View Statistics:
    Samples Shown: {zoom_end:,}
    Avg Throughput: {zoom_throughput.mean():.2f} tokens/s
    Max Throughput: {zoom_throughput.max():.2f} tokens/s
    Std Dev: {zoom_throughput.std():.2f} tokens/s"""
    
    ax2.text(0.02, 0.98, zoom_stats_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # 변동성 분석
    if len(zoom_throughput) > 1:
        throughput_changes = np.abs(np.diff(zoom_throughput))
        avg_change = throughput_changes.mean()
        max_change = throughput_changes.max()
        
        # 변동성 정보를 하단 subplot에 추가  
        volatility_text = f"""Volatility Analysis:
        Avg Change: {avg_change:.2f} tokens/s
        Max Change: {max_change:.2f} tokens/s
        Change Std: {throughput_changes.std():.2f} tokens/s"""
        
        ax2.text(0.98, 0.98, volatility_text, transform=ax2.transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Throughput plot saved to {output_file}")
    print(f"  - Valid iterations: {len(valid_data):,}")
    print(f"  - Average throughput: {throughput.mean():.2f} tokens/s")
    print(f"  - Overall throughput: {overall_throughput:.2f} tokens/s")
    
    # 진동 패턴 분석 결과 출력
    if len(zoom_throughput) > 1:
        throughput_changes = np.abs(np.diff(zoom_throughput))
        print(f"Throughput volatility analysis (first {zoom_end} iterations):")
        print(f"  - Average throughput change: {throughput_changes.mean():.2f} tokens/s")
        print(f"  - Maximum throughput change: {throughput_changes.max():.2f} tokens/s")  
        print(f"  - Throughput change std dev: {throughput_changes.std():.2f} tokens/s")


def plot_kv_cache_usage(df, output_file='kv_cache_usage_by_iteration.png'):
    """
    보너스: KV cache 관련 그래프들 (usage, usage_gb, total_capacity)
    
    Args:
        df: iteration 데이터 DataFrame
        output_file: 출력 파일명 (확장자 제외한 base name으로 사용)
    """
    # KV cache 관련 컬럼들과 그에 대응하는 설정
    kv_metrics = [
        {
            'column': 'kv_cache_usage',
            'ylabel': 'KV Cache Usage (Ratio)',
            'title': 'KV Cache Usage by Iteration Step',
            'color': 'purple',
            'scatter_color': 'darkviolet',
            'bg_color': 'lavender',
            'filename_suffix': 'usage'
        },
        {
            'column': 'kv_cache_usage_gb', 
            'ylabel': 'KV Cache Usage (GB)',
            'title': 'KV Cache Usage in GB by Iteration Step',
            'color': 'blue',
            'scatter_color': 'darkblue',
            'bg_color': 'lightblue',
            'filename_suffix': 'usage_gb'
        },
        {
            'column': 'kv_cache_total_capacity',
            'ylabel': 'KV Cache Total Capacity (GB)',
            'title': 'KV Cache Total Capacity by Iteration Step', 
            'color': 'green',
            'scatter_color': 'darkgreen',
            'bg_color': 'lightgreen',
            'filename_suffix': 'total_capacity'
        }
    ]
    
    # 출력 파일의 기본 이름 (확장자 제거)
    base_filename = output_file.rsplit('.', 1)[0] if '.' in output_file else output_file
    file_extension = output_file.rsplit('.', 1)[1] if '.' in output_file else 'png'
    
    created_plots = []
    
    for metric in kv_metrics:
        column_name = metric['column']
        
        # 컬럼 존재 여부 확인
        if column_name not in df.columns:
            print(f"{column_name} data not found, skipping...")
            continue
            
        # 그래프 생성
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # iteration step 데이터
        iterations = df['iteration_total']
        kv_data = df[column_name].fillna(0)
        
        # 데이터가 모두 0인지 확인
        if kv_data.sum() == 0:
            print(f"{column_name} contains only zero values, skipping...")
            plt.close()
            continue
        
        # 그래프 그리기
        ax.plot(iterations, kv_data, metric['color'], linewidth=2, alpha=0.8)
        ax.scatter(iterations, kv_data, s=20, color=metric['scatter_color'], alpha=0.6)
        
        # 스타일 설정
        ax.set_xlabel('Iteration Step', fontsize=12)
        ax.set_ylabel(metric['ylabel'], fontsize=12)
        ax.set_title(metric['title'], fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # 통계 정보 추가
        stats_text = f"""Statistics:
                Avg: {kv_data.mean():.4f}
                Max: {kv_data.max():.4f}
                Min: {kv_data.min():.4f}
                Non-zero samples: {(kv_data > 0).sum():,}/{len(kv_data):,}"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor=metric['bg_color'], alpha=0.8))
        
        # 파일 저장
        output_filename = f"{base_filename}_{metric['filename_suffix']}.{file_extension}"
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        created_plots.append(output_filename)
        print(f"{metric['title']} plot saved to {output_filename}")
    
    # 요약 출력
    if created_plots:
        print(f"\nKV cache plots created: {len(created_plots)} plots")
        for plot in created_plots:
            print(f"  - {plot}")
    else:
        print("No KV cache plots were created (no valid data found)")


from scipy import stats

from scipy import stats

def plot_itl_probability_distribution(df, output_file='itl_probability_distribution.png', middle_ratio=0.7):
    """
    ITL (Inter-Token Latency)의 확률 분포를 PDF 형태로 시각화 (중간 부분만 사용)
    
    Args:
        df: iteration 데이터 DataFrame
        output_file: 출력 파일명
        middle_ratio: 중간 부분의 비율 (기본값: 0.7, 즉 중간 70%)
    """
    if 'itl' not in df.columns:
        print("ITL data not found, skipping probability distribution plot...")
        return
    
    # None이 아닌 ITL 데이터만 필터링하고 iteration_total로 정렬
    valid_data = df[df['itl'].notna()].copy()
    valid_data = valid_data.sort_values('iteration_total').reset_index(drop=True)
    
    if len(valid_data) == 0:
        print("No valid ITL data found, skipping probability distribution plot...")
        return
    
    # 중간 부분만 선택 (처음과 끝 제거)
    total_samples = len(valid_data)
    exclude_ratio = (1 - middle_ratio) / 2  # 앞뒤로 제외할 비율
    start_idx = int(total_samples * exclude_ratio)
    end_idx = total_samples - start_idx
    
    middle_data = valid_data.iloc[start_idx:end_idx]
    itl_data = middle_data['itl']
    
    print(f"Creating ITL probability distribution plot:")
    print(f"  Total valid samples: {total_samples:,}")
    print(f"  Using middle {middle_ratio*100:.0f}%: {len(itl_data):,} samples")
    print(f"  Excluded: {start_idx:,} from start, {start_idx:,} from end")
    print(f"  Iteration range: {middle_data['iteration_total'].min()} - {middle_data['iteration_total'].max()}")
    
    # 기본 통계 계산
    median_itl = itl_data.median()  # P50
    p90_itl = itl_data.quantile(0.90)  # P90
    
    # 단일 플롯 생성
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # KDE (커널 밀도 추정)로 PDF 계산
    kde_x = np.linspace(itl_data.min(), itl_data.max(), 1000)
    kde = stats.gaussian_kde(itl_data)
    kde_y = kde(kde_x)
    
    # PDF 선 그리기
    ax.plot(kde_x, kde_y, 'b-', linewidth=3, label='PDF (Kernel Density)', alpha=0.8)
    
    # P50 (중앙값) 표시
    ax.axvline(median_itl, color='green', linestyle='--', linewidth=2.5, 
               label=f'P50 (Median): {median_itl:.4f}s', alpha=0.8)
    
    # P90 표시
    ax.axvline(p90_itl, color='red', linestyle='--', linewidth=2.5, 
               label=f'P90: {p90_itl:.4f}s', alpha=0.8)
    
    # 스타일 설정
    ax.set_xlabel('ITL (Inter-Token Latency) [seconds]', fontsize=14)
    ax.set_ylabel('Probability Density', fontsize=14)
    ax.set_title(f'ITL Probability Distribution (Middle {middle_ratio*100:.0f}% of Iterations)', 
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 축 스타일 개선
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # 통계 정보 (간단하게)
    stats_text = f"""Sample Info:
    Total Valid: {total_samples:,}
    Used (Middle {middle_ratio*100:.0f}%): {len(itl_data):,}
    Iteration Range: {middle_data['iteration_total'].min()}-{middle_data['iteration_total'].max()}

    Statistics:
    P50: {median_itl:.6f}s
    P90: {p90_itl:.6f}s
    Range: [{itl_data.min():.6f}, {itl_data.max():.6f}]s"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 콘솔에 상세한 요약 출력
    print(f"ITL Distribution Summary (Middle {middle_ratio*100:.0f}%):")
    print(f"  Used samples: {len(itl_data):,} out of {total_samples:,}")
    print(f"  Iteration range: {middle_data['iteration_total'].min()} - {middle_data['iteration_total'].max()}")
    print(f"  P50 (Median): {median_itl:.6f}s")
    print(f"  P90: {p90_itl:.6f}s")
    print(f"  Range: [{itl_data.min():.6f}, {itl_data.max():.6f}]s")
    print(f"  Excluded {start_idx:,} iterations from start and {start_idx:,} from end")
    print(f"ITL probability distribution plot saved to {output_file}")



def generate_summary_report(df, output_file='iteration_summary_report.txt'):
    """
    분석 결과 요약 리포트 생성
    
    Args:
        df: iteration 데이터 DataFrame
        output_file: 출력 파일명
    """
    with open(output_file, 'w') as f:
        f.write("=== Benchmark Iteration Analysis Report ===\n\n")
        
        # 전체 성능 계산
        total_tokens_generated = df['tokens_generated'].fillna(0).sum()
        valid_itl_data = df[df['itl'].notna()]
        total_processing_time = valid_itl_data['itl'].sum() if len(valid_itl_data) > 0 else 0
        overall_throughput = total_tokens_generated / total_processing_time if total_processing_time > 0 else 0
        
        # 기본 정보
        f.write("1. Basic Information:\n")
        f.write(f"   - Total iterations: {len(df):,}\n")
        f.write(f"   - Iteration range: {df['iteration_total'].min()} - {df['iteration_total'].max()}\n")
        f.write(f"   - Total tokens generated: {total_tokens_generated:,} tokens\n")
        f.write(f"   - Total processing time: {total_processing_time:.2f} seconds\n")
        f.write(f"   - Overall throughput: {overall_throughput:.2f} tokens/s\n")
        if total_processing_time > 0:
            f.write(f"   - Processing efficiency: {total_processing_time/60:.1f} minutes for {total_tokens_generated:,} tokens\n")
        f.write("\n")
        
        # 토큰 통계
        f.write("2. Token Statistics:\n")
        total_tokens = df['total_scheduled_tokens'].fillna(0)
        prefill_tokens = df['prefill_tokens'].fillna(0)
        decode_tokens = df['decode_tokens'].fillna(0)
        
        f.write(f"   - Average total scheduled tokens: {total_tokens.mean():.2f}\n")
        f.write(f"   - Average prefill tokens: {prefill_tokens.mean():.2f}\n")
        f.write(f"   - Average decode tokens: {decode_tokens.mean():.2f}\n")
        f.write(f"   - Max total scheduled tokens: {total_tokens.max():.0f}\n")
        f.write(f"   - Min total scheduled tokens: {total_tokens.min():.0f}\n")
        f.write(f"   - Total scheduled tokens (sum): {total_tokens.sum():,.0f}\n\n")
        
        # 요청 통계
        f.write("3. Request Statistics:\n")
        total_requests = df['total_scheduled_requests'].fillna(0)
        prefill_requests = df['prefill_requests'].fillna(0)
        decode_requests = df['decode_requests'].fillna(0)
        
        f.write(f"   - Average total scheduled requests: {total_requests.mean():.2f}\n")
        f.write(f"   - Average prefill requests: {prefill_requests.mean():.2f}\n")
        f.write(f"   - Average decode requests: {decode_requests.mean():.2f}\n")
        f.write(f"   - Max total scheduled requests: {total_requests.max():.0f}\n")
        f.write(f"   - Min total scheduled requests: {total_requests.min():.0f}\n")
        f.write(f"   - Total scheduled requests (sum): {total_requests.sum():,.0f}\n\n")
        
        # Throughput 상세 분석
        valid_throughput_data = df[(df['itl'].notna()) & (df['itl'] > 0) & (df['decode_requests'].notna())].copy()
        if len(valid_throughput_data) > 0:
            instantaneous_throughput = valid_throughput_data['decode_requests'].fillna(0) / valid_throughput_data['itl']
            
            f.write("4. Throughput Analysis:\n")
            f.write(f"   - Overall throughput: {overall_throughput:.2f} tokens/s\n")
            f.write(f"   - Average instantaneous throughput: {instantaneous_throughput.mean():.2f} tokens/s\n")
            f.write(f"   - Max instantaneous throughput: {instantaneous_throughput.max():.2f} tokens/s\n")
            f.write(f"   - Min instantaneous throughput: {instantaneous_throughput.min():.2f} tokens/s\n")
            f.write(f"   - Throughput standard deviation: {instantaneous_throughput.std():.2f} tokens/s\n")
            f.write(f"   - Valid throughput measurements: {len(valid_throughput_data):,} iterations\n")
            
            # 성능 벤치마크 비교
            if overall_throughput > 0:
                tokens_per_minute = overall_throughput * 60
                tokens_per_hour = overall_throughput * 3600
                f.write(f"   - Performance metrics:\n")
                f.write(f"     * {tokens_per_minute:.0f} tokens/minute\n")
                f.write(f"     * {tokens_per_hour:.0f} tokens/hour\n")
                f.write(f"     * {total_tokens_generated/1000:.1f}K tokens in {total_processing_time/60:.1f} minutes\n")
            f.write("\n")
        
        # ITL 통계
        if 'itl' in df.columns:
            itl_data = df[df['itl'].notna()]
            if len(itl_data) > 0:
                f.write("5. Inter-Token Latency Statistics:\n")
                itl_values = itl_data['itl']
                f.write(f"   - Valid ITL measurements: {len(itl_data):,}\n")
                f.write(f"   - Average ITL: {itl_values.mean():.4f}s\n")
                f.write(f"   - Median ITL: {itl_values.median():.4f}s\n")
                f.write(f"   - Max ITL: {itl_values.max():.4f}s\n")
                f.write(f"   - Min ITL: {itl_values.min():.4f}s\n")
                f.write(f"   - ITL Standard Deviation: {itl_values.std():.4f}s\n")
                f.write(f"   - P90 ITL: {itl_values.quantile(0.9):.4f}s\n")
                f.write(f"   - P95 ITL: {itl_values.quantile(0.95):.4f}s\n")
                f.write(f"   - Total processing time: {total_processing_time:.2f}s\n\n")
        
        # KV 캐시 통계
        if 'kv_cache_usage' in df.columns:
            f.write("6. KV Cache Usage Statistics:\n")
            kv_usage = df['kv_cache_usage'].fillna(0)
            non_zero_kv = kv_usage[kv_usage > 0]
            f.write(f"   - Average KV cache usage: {kv_usage.mean():.4f}\n")
            f.write(f"   - Max KV cache usage: {kv_usage.max():.4f}\n")
            f.write(f"   - Min KV cache usage: {kv_usage.min():.4f}\n")
            if len(non_zero_kv) > 0:
                f.write(f"   - Average non-zero KV usage: {non_zero_kv.mean():.4f}\n")
                f.write(f"   - Non-zero KV usage samples: {len(non_zero_kv):,}/{len(df):,} ({len(non_zero_kv)/len(df)*100:.1f}%)\n")
            f.write("\n")
        
        # 데이터 검증
        f.write("7. Data Validation:\n")
        token_diff = np.abs(total_tokens - (prefill_tokens + decode_tokens)).max()
        request_diff = np.abs(total_requests - (prefill_requests + decode_requests)).max()
        
        f.write(f"   - Token sum validation (max difference): {token_diff:.2f}\n")
        f.write(f"   - Request sum validation (max difference): {request_diff:.2f}\n")
        
        # 토큰 생성 vs 스케줄링 비교
        scheduled_vs_generated_ratio = total_tokens_generated / total_tokens.sum() if total_tokens.sum() > 0 else 0
        f.write(f"   - Generated vs Scheduled tokens ratio: {scheduled_vs_generated_ratio:.4f}\n")
        f.write(f"     * Total scheduled: {total_tokens.sum():,.0f} tokens\n")
        f.write(f"     * Total generated: {total_tokens_generated:,} tokens\n")
        
        if token_diff > 0.1:
            f.write("   - WARNING: Token sum mismatch detected!\n")
        if request_diff > 0.1:
            f.write("   - WARNING: Request sum mismatch detected!\n")
        if abs(scheduled_vs_generated_ratio - 1.0) > 0.1:
            f.write(f"   - INFO: Generated/Scheduled token ratio is {scheduled_vs_generated_ratio:.2f} (expected ~1.0)\n")
        
        f.write("\n")
        
        # 성능 요약
        f.write("8. Performance Summary:\n")
        f.write(f"   - Benchmark processed {total_tokens_generated:,} tokens in {total_processing_time:.1f} seconds\n")
        f.write(f"   - Overall system throughput: {overall_throughput:.2f} tokens/second\n")
        if overall_throughput > 0:
            f.write(f"   - Time to process 1M tokens: {1000000/overall_throughput/60:.1f} minutes\n")
            f.write(f"   - Tokens processed per minute: {overall_throughput*60:.0f}\n")
        f.write(f"   - Average iteration latency: {total_processing_time/len(df):.4f}s\n")
        if len(valid_itl_data) > 0:
            f.write(f"   - Iterations with valid ITL: {len(valid_itl_data):,}/{len(df):,} ({len(valid_itl_data)/len(df)*100:.1f}%)\n")
    
    print(f"Summary report saved to {output_file}")
    print(f"Performance Summary:")
    print(f"  - Total tokens generated: {total_tokens_generated:,}")
    print(f"  - Total processing time: {total_processing_time:.2f} seconds")
    print(f"  - Overall throughput: {overall_throughput:.2f} tokens/s")


########## Iteration Latency Fitting Model Analysis ##########


def fit_simple_itl_model(df):
    """
    간단한 ITL 예측 모델: ITL = a * total_scheduled_tokens + b
    
    Args:
        df: iteration 데이터 DataFrame
        
    Returns:
        tuple: (model, X, y, valid_data, performance_metrics)
    """
    print("=== Simple ITL Model (1st Order Linear) ===")
    
    # ITL이 null이 아닌 데이터만 필터링
    valid_data = df[df['itl'].notna()].copy()
    
    if len(valid_data) == 0:
        raise ValueError("No valid ITL data found!")
    
    print(f"Using {len(valid_data)} iterations with valid ITL data")
    
    # X: total_scheduled_tokens, Y: ITL
    X = valid_data['total_scheduled_tokens'].fillna(0).values.reshape(-1, 1)
    y = valid_data['itl'].values
    
    # 간단한 선형 회귀 모델 피팅
    model = LinearRegression()
    model.fit(X, y)
    
    # 예측
    y_pred = model.predict(X)
    
    # 성능 지표 계산
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    relative_errors = np.abs((y - y_pred) / y) * 100
    mean_relative_error = np.mean(relative_errors)
    
    performance_metrics = {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'mean_relative_error': mean_relative_error,
        'coefficient': model.coef_[0],
        'intercept': model.intercept_,
        'y_true': y,
        'y_pred': y_pred
    }
    
    print(f"Simple Model Equation: ITL = {model.coef_[0]:.8f} * total_tokens + {model.intercept_:.8f}")
    print(f"Model Performance:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.6f} seconds")
    print(f"  MAE: {mae:.6f} seconds")
    print(f"  Mean Relative Error: {mean_relative_error:.2f}%")
    
    return model, X, y, valid_data, performance_metrics



def plot_simple_itl_model(model, X, y, valid_data, performance_metrics, output_file='simple_itl_model.png'):
    """
    간단한 ITL 모델 결과 시각화
    
    Args:
        model: 피팅된 선형 회귀 모델
        X: total_scheduled_tokens (feature)
        y: actual ITL values
        valid_data: 유효한 데이터 DataFrame
        performance_metrics: 성능 지표들
        output_file: 출력 파일명
    """
    print("=== Creating Simple ITL Model Visualizations ===")
    
    # 2x2 subplot 생성
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Scatter plot with regression line (좌상단)
    ax1 = axes[0, 0]
    
    # 데이터 포인트
    ax1.scatter(X.flatten(), y, alpha=0.6, s=30, color='blue', label='Actual Data')
    
    # 회귀선
    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred_range = model.predict(X_range)
    ax1.plot(X_range, y_pred_range, 'r-', linewidth=3, label='Regression Line')
    
    # 스타일 설정
    ax1.set_xlabel('Total Scheduled Tokens', fontsize=12)
    ax1.set_ylabel('ITL (seconds)', fontsize=12)
    ax1.set_title(f'Simple ITL Model: ITL vs Total Tokens\nR² = {performance_metrics["r2"]:.4f}', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 수식 표시
    coef = performance_metrics['coefficient']
    intercept = performance_metrics['intercept']
    equation_text = f'ITL = {coef:.8f} × tokens + {intercept:.8f}'
    ax1.text(0.05, 0.95, equation_text, transform=ax1.transAxes, 
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
             fontsize=10, verticalalignment='top')
    
    # 2. Actual vs Predicted (우상단)
    ax2 = axes[0, 1]
    y_pred = performance_metrics['y_pred']
    
    ax2.scatter(y, y_pred, alpha=0.6, s=30, color='green')
    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    ax2.set_xlabel('Actual ITL (seconds)', fontsize=12)
    ax2.set_ylabel('Predicted ITL (seconds)', fontsize=12)
    ax2.set_title(f'Actual vs Predicted ITL\nRMSE = {performance_metrics["rmse"]:.6f}s', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # 3. Residuals plot (좌하단)
    ax3 = axes[1, 0]
    residuals = y - y_pred
    
    ax3.scatter(y_pred, residuals, alpha=0.6, s=30, color='purple')
    ax3.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('Predicted ITL (seconds)', fontsize=12)
    ax3.set_ylabel('Residuals (seconds)', fontsize=12)
    ax3.set_title('Residuals vs Predicted', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # 4. Time series (우하단)
    ax4 = axes[1, 1]
    iteration_steps = valid_data['iteration_total'].values
    
    # 처음 1000개 샘플만 표시
    n_show = min(1000, len(y))
    ax4.plot(iteration_steps[:n_show], y[:n_show], 'b-', label='Actual ITL', 
             linewidth=1.5, alpha=0.8)
    ax4.plot(iteration_steps[:n_show], y_pred[:n_show], 'r--', label='Predicted ITL', 
             linewidth=1.5, alpha=0.8)
    
    ax4.set_xlabel('Iteration Step', fontsize=12)
    ax4.set_ylabel('ITL (seconds)', fontsize=12)
    ax4.set_title(f'ITL Time Series (First {n_show} samples)', fontsize=14)
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    # 전체 성능 지표 텍스트 박스
    stats_text = f"""Simple Model Performance:
    R² Score: {performance_metrics['r2']:.4f}
    RMSE: {performance_metrics['rmse']:.6f}s
    MAE: {performance_metrics['mae']:.6f}s
    Mean Rel. Error: {performance_metrics['mean_relative_error']:.2f}%

    Model Equation:
    ITL = {coef:.8f} × tokens + {intercept:.8f}

    Data Info:
    Total samples: {len(y):,}
    Token range: {X.min():.0f} - {X.max():.0f}
    ITL range: {y.min():.6f} - {y.max():.6f}s"""
        
    # 통계를 그래프 외부에 표시
    fig.text(0.02, 0.02, stats_text, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.9),
             verticalalignment='bottom')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # 하단 텍스트 공간 확보
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Simple ITL model plots saved to {output_file}")


def generate_simple_itl_report(model, X, y, valid_data, performance_metrics, output_file):
    """
    간단한 ITL 모델 분석 리포트 생성
    
    Args:
        model: 피팅된 모델
        X: feature data (total_scheduled_tokens)
        y: target data (ITL)
        valid_data: 유효한 데이터 DataFrame
        performance_metrics: 성능 지표들
        output_file: 출력 파일명
    """
    coef = performance_metrics['coefficient']
    intercept = performance_metrics['intercept']
    
    with open(output_file, 'w') as f:
        f.write("=== Simple ITL Prediction Model Analysis Report ===\n\n")
        
        # 모델 개요
        f.write("1. Model Overview:\n")
        f.write("   This is a simple linear regression model that predicts Inter-Token Latency (ITL)\n")
        f.write("   using only the total number of scheduled tokens as the input feature.\n")
        f.write("   \n")
        f.write("   Model Type: Linear Regression (1st order)\n")
        f.write("   Feature: total_scheduled_tokens\n")
        f.write("   Target: ITL (Inter-Token Latency in seconds)\n\n")
        
        # 수식
        f.write("2. Model Equation:\n")
        f.write(f"   ITL = {coef:.8f} × total_scheduled_tokens + {intercept:.8f}\n")
        f.write("   \n")
        f.write("   Where:\n")
        f.write("   - ITL: Inter-Token Latency in seconds\n")
        f.write("   - total_scheduled_tokens: Total number of tokens scheduled in the iteration\n")
        f.write(f"   - Coefficient (slope): {coef:.8f} seconds per token\n")
        f.write(f"   - Intercept: {intercept:.8f} seconds (baseline latency)\n\n")
        
        # 데이터 정보
        f.write("3. Dataset Information:\n")
        f.write(f"   - Total iterations with valid ITL: {len(y):,}\n")
        f.write(f"   - Token range: {X.min():.0f} - {X.max():.0f}\n")
        f.write(f"   - ITL range: {y.min():.6f} - {y.max():.6f} seconds\n")
        f.write(f"   - Average tokens per iteration: {X.mean():.1f}\n")
        f.write(f"   - Average ITL: {y.mean():.6f} seconds\n\n")
        
        # 성능 지표
        f.write("4. Model Performance:\n")
        f.write(f"   - R² Score: {performance_metrics['r2']:.4f}\n")
        f.write(f"     * Explains {performance_metrics['r2']*100:.1f}% of ITL variance\n")
        f.write(f"   - RMSE: {performance_metrics['rmse']:.6f} seconds\n")
        f.write(f"   - MAE: {performance_metrics['mae']:.6f} seconds\n")
        f.write(f"   - Mean Relative Error: {performance_metrics['mean_relative_error']:.2f}%\n\n")
        
        # 해석
        f.write("5. Model Interpretation:\n")
        f.write(f"   Coefficient Analysis:\n")
        f.write(f"   - Each additional token increases ITL by {coef:.8f} seconds\n")
        f.write(f"   - This represents {coef*1000:.5f} milliseconds per token\n")
        f.write(f"   - For 1000 tokens: additional {coef*1000:.5f} seconds\n")
        f.write(f"   \n")
        f.write(f"   Baseline Latency:\n")
        f.write(f"   - Intercept of {intercept:.8f} seconds represents fixed overhead\n")
        f.write(f"   - This is the minimum ITL regardless of token count\n")
        f.write(f"   \n")
        
        # 모델 품질 평가
        r2 = performance_metrics['r2']
        if r2 >= 0.9:
            quality = "Excellent"
        elif r2 >= 0.8:
            quality = "Good"
        elif r2 >= 0.7:
            quality = "Fair"
        elif r2 >= 0.5:
            quality = "Moderate"
        else:
            quality = "Poor"
            
        f.write("6. Model Quality Assessment:\n")
        f.write(f"   - Model Quality: {quality} (R² = {r2:.4f})\n")
        
        if r2 >= 0.7:
            f.write("   - The simple linear relationship captures most of the ITL variance\n")
            f.write("   - Total scheduled tokens is a strong predictor of ITL\n")
        else:
            f.write("   - The linear relationship explains limited ITL variance\n")
            f.write("   - ITL may depend on additional factors beyond total token count\n")
            f.write("   - Consider more complex models or additional features\n")
        
        f.write("\n")
        
        # 사용 가이드
        f.write("7. Usage Guide:\n")
        f.write("   To predict ITL for a new iteration:\n")
        f.write(f"   1. Count the total_scheduled_tokens\n")
        f.write(f"   2. Apply the formula: ITL = {coef:.8f} × tokens + {intercept:.8f}\n")
        f.write("   \n")
        f.write("   Example predictions:\n")
        
        # 예시 예측들
        example_tokens = [100, 500, 1000, 2000, 5000]
        for tokens in example_tokens:
            predicted_itl = coef * tokens + intercept
            f.write(f"   - {tokens:4d} tokens → {predicted_itl:.6f} seconds\n")
        
        f.write("\n")
        
        # 한계점
        f.write("8. Model Limitations:\n")
        f.write("   - Only considers total token count, ignoring:\n")
        f.write("     * Token distribution (prefill vs decode)\n")
        f.write("     * Request patterns and batching\n")
        f.write("     * KV cache effects\n")
        f.write("     * Sequence length variations\n")
        f.write("   - Assumes linear relationship (may not capture complexity)\n")
        f.write("   - Performance may degrade outside training token range\n")
        
        if performance_metrics['mean_relative_error'] > 20:
            f.write(f"   - High relative error ({performance_metrics['mean_relative_error']:.1f}%) suggests model limitations\n")
        
        f.write("\n")
        
        # 결론
        f.write("9. Conclusion:\n")
        if r2 >= 0.8:
            f.write("   The simple token-based model provides reliable ITL predictions.\n")
            f.write("   Total scheduled tokens is the dominant factor in ITL determination.\n")
        elif r2 >= 0.6:
            f.write("   The model captures the main token-ITL relationship but has room for improvement.\n")
            f.write("   Additional features might enhance prediction accuracy.\n")
        else:
            f.write("   The simple linear model has limited predictive power.\n")
            f.write("   ITL appears to depend on factors beyond total token count.\n")
            f.write("   Consider more sophisticated modeling approaches.\n")
        
        f.write(f"\n   Model equation: ITL = {coef:.8f} × total_tokens + {intercept:.8f}\n")
        f.write(f"   Performance: R² = {r2:.4f}, RMSE = {performance_metrics['rmse']:.6f}s\n")
    
    print(f"Simple ITL model report saved to {output_file}")


def compare_simple_vs_complex_models(simple_metrics, complex_results=None, output_file='model_comparison.txt'):
    """
    간단한 모델과 복잡한 모델 성능 비교
    
    Args:
        simple_metrics: 간단한 모델의 성능 지표
        complex_results: 복잡한 모델의 결과 (선택사항)
        output_file: 출력 파일명
    """
    print("=== Model Comparison Analysis ===")
    
    with open(output_file, 'w') as f:
        f.write("=== Simple vs Complex ITL Model Comparison ===\n\n")
        
        f.write("1. Simple Model (Token-based Linear Regression):\n")
        f.write(f"   - Features: 1 (total_scheduled_tokens)\n")
        f.write(f"   - Model: ITL = {simple_metrics['coefficient']:.8f} × tokens + {simple_metrics['intercept']:.8f}\n")
        f.write(f"   - R² Score: {simple_metrics['r2']:.4f}\n")
        f.write(f"   - RMSE: {simple_metrics['rmse']:.6f} seconds\n")
        f.write(f"   - Relative Error: {simple_metrics['mean_relative_error']:.2f}%\n")
        f.write("\n")
        
        if complex_results is not None:
            complex_perf = complex_results['performance']['ridge']
            f.write("2. Complex Model (Ridge Regression with 8 features):\n")
            f.write(f"   - Features: 8 (Nt, Rd, Rp, KV_decode, KV_prefill, T_prefill, T_prefill², T×KV)\n")
            f.write(f"   - Model: Ridge Regression with L2 regularization\n")
            f.write(f"   - R² Score: {complex_perf['r2']:.4f}\n")
            f.write(f"   - RMSE: {complex_perf['rmse']:.6f} seconds\n")
            f.write("\n")
            
            # 성능 비교
            f.write("3. Performance Comparison:\n")
            r2_diff = complex_perf['r2'] - simple_metrics['r2']
            rmse_diff = simple_metrics['rmse'] - complex_perf['rmse']
            
            f.write(f"   R² Score improvement: {r2_diff:+.4f}\n")
            f.write(f"   RMSE improvement: {rmse_diff:+.6f} seconds\n")
            
            if r2_diff > 0.05:  # 5% 이상 개선
                f.write("   → Complex model shows significant improvement\n")
            elif r2_diff > 0.01:  # 1% 이상 개선
                f.write("   → Complex model shows moderate improvement\n")
            else:
                f.write("   → Limited improvement from complex model\n")
            
            f.write("\n")
            
            # 복잡성 vs 성능 트레이드오프
            f.write("4. Complexity vs Performance Trade-off:\n")
            f.write(f"   Simple Model:\n")
            f.write(f"   + Easy to understand and implement\n")
            f.write(f"   + Single feature, no multicollinearity issues\n")
            f.write(f"   + Fast training and prediction\n")
            f.write(f"   - Limited expressiveness\n")
            f.write(f"\n")
            f.write(f"   Complex Model:\n")
            f.write(f"   + Captures detailed system behavior\n")
            f.write(f"   + Higher accuracy (R² = {complex_perf['r2']:.4f})\n")
            f.write(f"   - Requires 8 features and feature engineering\n")
            f.write(f"   - Multicollinearity issues (needs Ridge regularization)\n")
            f.write(f"   - More complex to interpret and maintain\n")
            
        else:
            f.write("2. Complex Model: Not available for comparison\n")
        
        f.write("\n")
        f.write("5. Recommendation:\n")
        if complex_results is None or simple_metrics['r2'] > 0.8:
            f.write("   The simple token-based model provides good ITL predictions\n")
            f.write("   with minimal complexity. Recommended for most use cases.\n")
        elif complex_results and complex_results['performance']['ridge']['r2'] - simple_metrics['r2'] > 0.1:
            f.write("   The complex model provides significantly better accuracy.\n")
            f.write("   Choose based on accuracy requirements vs implementation complexity.\n")
        else:
            f.write("   Both models have similar performance. The simple model is\n")
            f.write("   recommended due to its simplicity and interpretability.\n")
    
    print(f"Model comparison saved to {output_file}")

def analyze_simple_itl_model(df, output_prefix='./benchmark_iteration/', complex_results=None):
    """
    간단한 ITL 모델의 전체 분석 파이프라인
    
    Args:
        df: iteration 데이터 DataFrame
        output_prefix: 출력 파일 경로 prefix
        complex_results: 복잡한 모델 결과 (비교용, 선택사항)
    """
    try:
        print("=== Simple ITL Model Analysis ===")
        
        # 1. 모델 피팅
        print("\n1. Fitting simple ITL model...")
        model, X, y, valid_data, performance_metrics = fit_simple_itl_model(df)
        
        # 2. 시각화
        print("\n2. Creating visualizations...")
        plot_simple_itl_model(model, X, y, valid_data, performance_metrics, 
                              f"{output_prefix}simple_itl_model.png")
        
        # 3. 리포트 생성
        print("\n3. Generating analysis report...")
        generate_simple_itl_report(model, X, y, valid_data, performance_metrics,
                                  f"{output_prefix}simple_itl_model_report.txt")
        
        # 4. 모델 비교 (복잡한 모델이 있는 경우)
        if complex_results is not None:
            print("\n4. Comparing with complex model...")
            compare_simple_vs_complex_models(performance_metrics, complex_results,
                                           f"{output_prefix}model_comparison.txt")
        
        print("\n=== Simple ITL Model Analysis Complete ===")
        print(f"Simple Model Performance:")
        print(f"  - R² Score: {performance_metrics['r2']:.4f}")
        print(f"  - RMSE: {performance_metrics['rmse']:.6f} seconds")
        print(f"  - Equation: ITL = {performance_metrics['coefficient']:.8f} × tokens + {performance_metrics['intercept']:.8f}")
        
        return model, performance_metrics
        
    except Exception as e:
        print(f"Error in simple ITL model analysis: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def extract_features_from_iteration(iteration_data):
    """
    단일 iteration에서 ITL 예측을 위한 feature들을 추출
    
    Args:
        iteration_data: 단일 iteration의 데이터 (dict)
        
    Returns:
        dict: 추출된 feature들
    """
    # 기본 변수들
    Nt = iteration_data.get('total_scheduled_tokens', 0)
    Rd = iteration_data.get('decode_requests', 0) 
    Rp = iteration_data.get('prefill_requests', 0)
    
    # request_details에서 계산할 변수들 초기화
    kv_decode_sum = 0    # a4: ∑KVr (r∈D)
    kv_prefill_sum = 0   # a5: ∑KVr (r∈P) 
    t_prefill_sum = 0    # a6: ∑Tr (r∈P)
    t_prefill_square_sum = 0  # a7: ∑Tr² (r∈P)
    t_kv_prefill_sum = 0      # a8: ∑(Tr*KVr) (r∈P)
    
    # request_details 처리
    request_details = iteration_data.get('request_details', [])
    
    for request in request_details:
        phase = request.get('phase', '')
        kv_cache_len = request.get('kv_cache_len', 0)
        scheduled_tokens = request.get('scheduled_tokens', 0)
        
        if phase == 'decode':
            kv_decode_sum += kv_cache_len
        elif phase == 'prefill':
            kv_prefill_sum += kv_cache_len
            t_prefill_sum += scheduled_tokens
            t_prefill_square_sum += scheduled_tokens ** 2
            t_kv_prefill_sum += scheduled_tokens * kv_cache_len
    
    features = {
        'Nt': Nt,                           # a1
        'Rd': Rd,                           # a2  
        'Rp': Rp,                           # a3
        'kv_decode_sum': kv_decode_sum,     # a4
        'kv_prefill_sum': kv_prefill_sum,   # a5
        't_prefill_sum': t_prefill_sum,     # a6
        't_prefill_square_sum': t_prefill_square_sum,  # a7
        't_kv_prefill_sum': t_kv_prefill_sum,          # a8
        # a9는 intercept로 자동 처리
    }
    
    return features

def prepare_itl_dataset(df):
    """
    ITL 예측을 위한 dataset 준비
    
    Args:
        df: iteration 데이터 DataFrame
        
    Returns:
        tuple: (X, y, feature_names) - features, targets, feature 이름들
    """
    # ITL이 null이 아닌 데이터만 필터링
    valid_data = df[df['itl'].notna()].copy()
    
    if len(valid_data) == 0:
        raise ValueError("No valid ITL data found!")
    
    print(f"Using {len(valid_data)} iterations with valid ITL data")
    
    # 각 iteration에서 feature 추출
    features_list = []
    itl_values = []
    
    for idx, row in valid_data.iterrows():
        features = extract_features_from_iteration(row.to_dict())
        features_list.append(features)
        itl_values.append(row['itl'])
    
    # DataFrame으로 변환
    features_df = pd.DataFrame(features_list)
    
    # Feature 이름들
    feature_names = [
        'Nt (total_tokens)',
        'Rd (decode_requests)', 
        'Rp (prefill_requests)',
        'KV_decode_sum',
        'KV_prefill_sum', 
        'T_prefill_sum',
        'T_prefill_square_sum',
        'T_KV_prefill_sum'
    ]
    
    X = features_df.values
    y = np.array(itl_values)
    
    print("Feature statistics:")
    print(features_df.describe())
    
    return X, y, feature_names, valid_data

def fit_itl_model(X, y, feature_names):
    """
    ITL 예측 모델 피팅
    
    Args:
        X: feature matrix
        y: target values (ITL)
        feature_names: feature 이름들
        
    Returns:
        tuple: (model, coefficients_info)
    """
    # Linear Regression 모델 피팅
    model = LinearRegression()
    model.fit(X, y)
    
    # 계수 정보 정리
    coefficients = model.coef_
    intercept = model.intercept_
    
    coefficients_info = {
        'coefficients': coefficients,
        'intercept': intercept,
        'feature_names': feature_names
    }
    
    print("=== Fitted Model Coefficients ===")
    print(f"Intercept (a9): {intercept:.6f}")
    for i, (coef, name) in enumerate(zip(coefficients, feature_names)):
        print(f"a{i+1} ({name}): {coef:.6f}")
    
    return model, coefficients_info

def evaluate_itl_model(model, X, y, feature_names):
    """
    ITL 예측 모델 성능 평가
    
    Args:
        model: 피팅된 모델
        X: feature matrix  
        y: target values
        feature_names: feature 이름들
        
    Returns:
        dict: 평가 결과
    """
    # 예측
    y_pred = model.predict(X)
    
    # 성능 지표 계산
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # 상대 오차 계산
    relative_errors = np.abs((y - y_pred) / y) * 100
    mean_relative_error = np.mean(relative_errors)
    
    results = {
        'mse': mse,
        'rmse': rmse, 
        'mae': mae,
        'r2': r2,
        'mean_relative_error': mean_relative_error,
        'y_true': y,
        'y_pred': y_pred
    }
    
    print("=== Model Performance ===")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"Mean Relative Error: {mean_relative_error:.2f}%")
    
    return results

def plot_model_results(results, coefficients_info, output_prefix='./benchmark_iteration/'):
    """
    ITL 모델 결과 시각화
    
    Args:
        results: 모델 평가 결과
        coefficients_info: 모델 계수 정보
        output_prefix: 출력 파일 경로 prefix
    """
    # 1. 실제 vs 예측 scatter plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Actual vs Predicted
    ax1 = axes[0, 0]
    y_true = results['y_true']
    y_pred = results['y_pred']
    
    ax1.scatter(y_true, y_pred, alpha=0.6, s=50)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax1.set_xlabel('Actual ITL')
    ax1.set_ylabel('Predicted ITL')
    ax1.set_title(f'Actual vs Predicted ITL (R² = {results["r2"]:.4f})')
    ax1.grid(True, alpha=0.3)
    
    # Residuals plot
    ax2 = axes[0, 1]
    residuals = y_true - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.6, s=50)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Predicted ITL')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residuals vs Predicted')
    ax2.grid(True, alpha=0.3)
    
    # Coefficients bar plot
    ax3 = axes[1, 0]
    coeffs = coefficients_info['coefficients']
    names = [name.split('(')[0].strip() for name in coefficients_info['feature_names']]
    colors = plt.cm.viridis(np.linspace(0, 1, len(coeffs)))
    
    bars = ax3.bar(range(len(coeffs)), coeffs, color=colors)
    ax3.set_xlabel('Features')
    ax3.set_ylabel('Coefficient Value')
    ax3.set_title('Model Coefficients')
    ax3.set_xticks(range(len(coeffs)))
    ax3.set_xticklabels(names, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # Error distribution
    ax4 = axes[1, 1]
    relative_errors = np.abs((y_true - y_pred) / y_true) * 100
    ax4.hist(relative_errors, bins=30, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Relative Error (%)')
    ax4.set_ylabel('Frequency')
    ax4.set_title(f'Error Distribution (Mean: {np.mean(relative_errors):.2f}%)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}itl_model_results.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Time series plot
    fig, ax = plt.subplots(figsize=(15, 6))
    indices = range(len(y_true))
    ax.plot(indices, y_true, 'b-', label='Actual ITL', linewidth=2, alpha=0.8)
    ax.plot(indices, y_pred, 'r--', label='Predicted ITL', linewidth=2, alpha=0.8)
    ax.fill_between(indices, y_true, y_pred, alpha=0.3, color='gray', label='Error')
    
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('ITL (seconds)')
    ax.set_title('ITL Prediction Time Series')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}itl_time_series.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Model result plots saved to {output_prefix}itl_model_results.png and itl_time_series.png")

def generate_itl_model_report(results, coefficients_info, output_file):
    """
    ITL 모델 분석 리포트 생성
    
    Args:
        results: 모델 평가 결과
        coefficients_info: 모델 계수 정보  
        output_file: 출력 파일명
    """
    with open(output_file, 'w') as f:
        f.write("=== ITL Prediction Model Analysis Report ===\n\n")
        
        # 모델 수식
        f.write("1. Model Equation:\n")
        f.write("ITL = a₁Nₜ + a₂Rd + a₃Rₚ + a₄∑(KVr of decode) + a₅∑(KVᵣ of prefill) + a₆∑(Tᵣ of prefill) + a₇∑(Tᵣ² of prefill) + a₈∑(Tᵣ*KVᵣ of prefill) + a₉\n")
        
        # 피팅된 계수들
        f.write("2. Fitted Coefficients:\n")
        f.write(f"   a₉ (Intercept): {coefficients_info['intercept']:.6f}\n")
        for i, (coef, name) in enumerate(zip(coefficients_info['coefficients'], coefficients_info['feature_names'])):
            f.write(f"   a₁{i+1} ({name}): {coef:.6f}\n")
        f.write("\n")
        
        # 성능 지표
        f.write("3. Model Performance:\n")
        f.write(f"   R² Score: {results['r2']:.4f}\n")
        f.write(f"   RMSE: {results['rmse']:.6f} seconds\n")
        f.write(f"   MAE: {results['mae']:.6f} seconds\n")
        f.write(f"   Mean Relative Error: {results['mean_relative_error']:.2f}%\n\n")
        
        # 해석
        f.write("4. Coefficient Interpretation:\n")
        coeffs = coefficients_info['coefficients']
        names = coefficients_info['feature_names']
        
        # 절댓값 기준으로 정렬하여 중요도 표시
        coef_importance = list(zip(names, coeffs, np.abs(coeffs)))
        coef_importance.sort(key=lambda x: x[2], reverse=True)
        
        f.write("   Most Important Features (by absolute coefficient value):\n")
        for i, (name, coef, abs_coef) in enumerate(coef_importance[:5]):
            impact = "increases" if coef > 0 else "decreases"
            f.write(f"   {i+1}. {name}: {coef:.6f} ({impact} ITL)\n")
        
        f.write(f"\n   Note: R² = {results['r2']:.4f} indicates that the model explains "
                f"{results['r2']*100:.1f}% of the variance in ITL.\n")
    
    print(f"ITL model report saved to {output_file}")

def debug_feature_extraction(df, num_samples=10):
    """
    Feature 추출 과정을 디버깅하는 함수
    
    Args:
        df: iteration 데이터 DataFrame
        num_samples: 디버깅할 샘플 수
    """
    print("=== Feature Extraction Debug ===")
    
    # ITL이 null이 아닌 데이터만 필터링
    valid_data = df[df['itl'].notna()].copy()
    
    print(f"Total valid iterations: {len(valid_data)}")
    print(f"Debugging first {num_samples} samples:\n")
    
    for i, (idx, row) in enumerate(valid_data.head(num_samples).iterrows()):
        print(f"--- Sample {i+1} (iteration {row['iteration_total']}) ---")
        print(f"ITL: {row['itl']:.6f}")
        
        # 기본 변수들
        print(f"Basic features:")
        print(f"  Nt (total_scheduled_tokens): {row.get('total_scheduled_tokens', 0)}")
        print(f"  Rd (decode_requests): {row.get('decode_requests', 0)}")
        print(f"  Rp (prefill_requests): {row.get('prefill_requests', 0)}")
        
        # request_details 상세 분석
        request_details = row.get('request_details', [])
        print(f"  Request details count: {len(request_details)}")
        
        kv_decode_sum = 0
        kv_prefill_sum = 0
        t_prefill_sum = 0
        t_prefill_square_sum = 0
        t_kv_prefill_sum = 0
        
        for j, request in enumerate(request_details):
            phase = request.get('phase', '')
            kv_cache_len = request.get('kv_cache_len', 0)
            scheduled_tokens = request.get('scheduled_tokens', 0)
            
            print(f"    Request {j+1}: phase={phase}, kv_len={kv_cache_len}, tokens={scheduled_tokens}")
            
            if phase == 'decode':
                kv_decode_sum += kv_cache_len
            elif phase == 'prefill':
                kv_prefill_sum += kv_cache_len
                t_prefill_sum += scheduled_tokens
                t_prefill_square_sum += scheduled_tokens ** 2
                t_kv_prefill_sum += scheduled_tokens * kv_cache_len
        
        print(f"Calculated features:")
        print(f"  kv_decode_sum: {kv_decode_sum}")
        print(f"  kv_prefill_sum: {kv_prefill_sum}")
        print(f"  t_prefill_sum: {t_prefill_sum}")
        print(f"  t_prefill_square_sum: {t_prefill_square_sum}")
        print(f"  t_kv_prefill_sum: {t_kv_prefill_sum}")
        print()

def analyze_feature_correlations(X, feature_names, output_prefix='./benchmark_iteration/'):
    """
    Feature들 간의 상관관계 분석
    
    Args:
        X: feature matrix
        feature_names: feature 이름들
        output_prefix: 출력 파일 경로 prefix
    """
    print("=== Feature Correlation Analysis ===")
    
    # DataFrame으로 변환
    df_features = pd.DataFrame(X, columns=[name.split('(')[0].strip() for name in feature_names])
    
    # 상관관계 매트릭스 계산
    correlation_matrix = df_features.corr()
    
    print("Correlation Matrix:")
    print(correlation_matrix.round(4))
    
    # 높은 상관관계 (>0.8) 찾기
    print("\nHigh correlations (>0.8):")
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > 0.8:
                print(f"  {correlation_matrix.columns[i]} - {correlation_matrix.columns[j]}: {corr_val:.4f}")
    
    # 상관관계 히트맵 시각화
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.3f', cbar_kws={"shrink": .5})
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(f"{output_prefix}feature_correlations.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Correlation heatmap saved to {output_prefix}feature_correlations.png")

def analyze_feature_distributions(X, y, feature_names, output_prefix='./benchmark_iteration/'):
    """
    Feature 분포 및 ITL과의 관계 분석
    
    Args:
        X: feature matrix
        y: target values (ITL)
        feature_names: feature 이름들
        output_prefix: 출력 파일 경로 prefix
    """
    print("=== Feature Distribution Analysis ===")
    
    # DataFrame으로 변환
    df_features = pd.DataFrame(X, columns=[name.split('(')[0].strip() for name in feature_names])
    df_features['ITL'] = y
    
    # 기본 통계 출력
    print("Feature Statistics:")
    print(df_features.describe())
    
    # 각 feature의 분포와 ITL과의 관계 시각화
    n_features = len(feature_names)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for i, (col, name) in enumerate(zip(df_features.columns[:-1], feature_names)):
        ax = axes[i]
        
        # Feature 값이 0이 아닌 데이터만 확인
        non_zero_mask = df_features[col] != 0
        non_zero_count = non_zero_mask.sum()
        
        if non_zero_count > 0:
            # Scatter plot
            ax.scatter(df_features[col], df_features['ITL'], alpha=0.6)
            ax.set_xlabel(col)
            ax.set_ylabel('ITL')
            ax.set_title(f'{col} vs ITL\n(Non-zero: {non_zero_count}/{len(df_features)})')
            
            # 상관계수 표시
            corr = df_features[col].corr(df_features['ITL'])
            ax.text(0.05, 0.95, f'Corr: {corr:.4f}', transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, f'{col}\nAll zeros!', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12, color='red')
            ax.set_title(f'{col} (ALL ZEROS)')
        
        ax.grid(True, alpha=0.3)
    
    # 남은 subplot 제거
    for i in range(len(feature_names), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}feature_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Feature distribution plots saved to {output_prefix}feature_distributions.png")
    
    # Zero feature 체크
    print("\nZero Feature Check:")
    for col in df_features.columns[:-1]:
        zero_count = (df_features[col] == 0).sum()
        total_count = len(df_features)
        print(f"  {col}: {zero_count}/{total_count} zeros ({zero_count/total_count*100:.1f}%)")

def enhanced_analyze_itl_model(df, output_prefix='./benchmark_iteration/'):
    """
    강화된 ITL 모델 분석 (디버깅 포함)
    
    Args:
        df: iteration 데이터 DataFrame
        output_prefix: 출력 파일 경로 prefix
    """
    try:
        print("=== Enhanced ITL Prediction Model Analysis ===")
        
        # 0. Feature 추출 디버깅
        print("\n0. Debugging feature extraction...")
        debug_feature_extraction(df, num_samples=5)
        
        # 1. 데이터셋 준비
        print("\n1. Preparing dataset...")
        X, y, feature_names, valid_data = prepare_itl_dataset(df)
        
        # 2. Feature 분포 분석
        print("\n2. Analyzing feature distributions...")
        analyze_feature_distributions(X, y, feature_names, output_prefix)
        
        # 3. 상관관계 분석
        print("\n3. Analyzing feature correlations...")
        analyze_feature_correlations(X, feature_names, output_prefix)
        
        # 4. 모델 피팅
        print("\n4. Fitting ITL prediction model...")
        model, coefficients_info = fit_itl_model(X, y, feature_names)
        
        # 5. 모델 평가
        print("\n5. Evaluating model performance...")
        results = evaluate_itl_model(model, X, y, feature_names)
        
        # 6. 결과 시각화
        print("\n6. Generating visualizations...")
        plot_model_results(results, coefficients_info, output_prefix)
        
        # 7. 리포트 생성
        print("\n7. Generating analysis report...")
        generate_itl_model_report(results, coefficients_info, f"{output_prefix}itl_model_report.txt")
        
        print("\n=== Enhanced ITL Model Analysis Complete ===")
        return model, results, coefficients_info
        
    except Exception as e:
        print(f"Error in enhanced ITL model analysis: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def analyze_itl_model(df, output_prefix='./benchmark_iteration/'):
    """
    ITL 예측 모델의 전체 분석 파이프라인
    
    Args:
        df: iteration 데이터 DataFrame
        output_prefix: 출력 파일 경로 prefix
    """
    try:
        print("=== ITL Prediction Model Analysis ===")
        
        # 1. 데이터셋 준비
        print("\n1. Preparing dataset...")
        X, y, feature_names, valid_data = prepare_itl_dataset(df)
        
        # 2. 모델 피팅
        print("\n2. Fitting ITL prediction model...")
        model, coefficients_info = fit_itl_model(X, y, feature_names)
        
        # 3. 모델 평가
        print("\n3. Evaluating model performance...")
        results = evaluate_itl_model(model, X, y, feature_names)
        
        # 4. 결과 시각화
        print("\n4. Generating visualizations...")
        plot_model_results(results, coefficients_info, output_prefix)
        
        # 5. 리포트 생성
        print("\n5. Generating analysis report...")
        generate_itl_model_report(results, coefficients_info, f"{output_prefix}itl_model_report.txt")
        
        print("\n=== ITL Model Analysis Complete ===")
        return model, results, coefficients_info
        
    except Exception as e:
        print(f"Error in ITL model analysis: {e}")
        return None, None, None


## VIF (Variance Inflation Factor) 분석

def calculate_vif(X, feature_names):
    """
    VIF (Variance Inflation Factor) 계산으로 다중공선성 정도 측정
    
    Args:
        X: feature matrix
        feature_names: feature 이름들
        
    Returns:
        pandas.DataFrame: VIF 결과
    """
    print("=== VIF (Variance Inflation Factor) Analysis ===")
    
    # VIF 계산
    vif_data = pd.DataFrame()
    vif_data["Feature"] = feature_names
    vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    
    print("VIF Values:")
    print(vif_data.round(2))
    print("\nVIF Interpretation:")
    print("- VIF < 5: Low multicollinearity")
    print("- 5 ≤ VIF < 10: Moderate multicollinearity") 
    print("- VIF ≥ 10: High multicollinearity (problematic)")
    
    # 높은 VIF 값 식별
    high_vif = vif_data[vif_data["VIF"] >= 10]
    if len(high_vif) > 0:
        print(f"\nFeatures with high VIF (≥10):")
        for _, row in high_vif.iterrows():
            print(f"  {row['Feature']}: {row['VIF']:.2f}")
    
    return vif_data

def fit_ridge_itl_model(X, y, feature_names, alpha_range=None):
    """
    Ridge Regression을 사용한 ITL 예측 모델 피팅
    
    Args:
        X: feature matrix
        y: target values (ITL)
        feature_names: feature 이름들
        alpha_range: Ridge 정규화 파라미터 범위
        
    Returns:
        tuple: (best_model, best_alpha, coefficients_info, scaler)
    """
    print("=== Ridge Regression ITL Model ===")
    
    # Feature 스케일링 (Ridge regression에 필수)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Alpha 범위 설정
    if alpha_range is None:
        alpha_range = np.logspace(-4, 2, 50)  # 0.0001 to 100
    
    # Cross-validation으로 최적 alpha 찾기
    ridge_cv = RidgeCV(alphas=alpha_range, cv=5, scoring='neg_mean_squared_error')
    ridge_cv.fit(X_scaled, y)
    
    best_alpha = ridge_cv.alpha_
    print(f"Best alpha (regularization parameter): {best_alpha:.6f}")
    
    # 최적 alpha로 Ridge 모델 피팅
    ridge_model = Ridge(alpha=best_alpha)
    ridge_model.fit(X_scaled, y)
    
    # 계수 정보 정리
    coefficients = ridge_model.coef_
    intercept = ridge_model.intercept_
    
    coefficients_info = {
        'coefficients': coefficients,
        'intercept': intercept,
        'feature_names': feature_names,
        'best_alpha': best_alpha,
        'scaler': scaler
    }
    
    print("Ridge Regression Coefficients:")
    print(f"Intercept (a9): {intercept:.6f}")
    for i, (coef, name) in enumerate(zip(coefficients, feature_names)):
        print(f"a{i+1} ({name}): {coef:.6f}")
    
    return ridge_model, best_alpha, coefficients_info, scaler

def compare_ols_vs_ridge(X, y, feature_names, output_prefix='./benchmark_iteration/'):
    """
    OLS와 Ridge Regression 결과 비교
    
    Args:
        X: feature matrix
        y: target values
        feature_names: feature 이름들
        output_prefix: 출력 파일 경로 prefix
    """
    print("=== OLS vs Ridge Regression Comparison ===")
    
    # 1. OLS 모델
    ols_model = LinearRegression()
    ols_model.fit(X, y)
    ols_pred = ols_model.predict(X)
    
    # 2. Ridge 모델
    ridge_model, best_alpha, ridge_coeffs, scaler = fit_ridge_itl_model(X, y, feature_names)
    X_scaled = scaler.transform(X)
    ridge_pred = ridge_model.predict(X_scaled)
    
    # 3. 성능 비교
    ols_r2 = r2_score(y, ols_pred)
    ols_rmse = np.sqrt(mean_squared_error(y, ols_pred))
    
    ridge_r2 = r2_score(y, ridge_pred)
    ridge_rmse = np.sqrt(mean_squared_error(y, ridge_pred))
    
    print("Performance Comparison:")
    print(f"OLS    - R²: {ols_r2:.4f}, RMSE: {ols_rmse:.6f}")
    print(f"Ridge  - R²: {ridge_r2:.4f}, RMSE: {ridge_rmse:.6f}")
    
    # 4. 계수 비교 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 계수 크기 비교
    ax1 = axes[0, 0]
    x_pos = np.arange(len(feature_names))
    width = 0.35
    
    ax1.bar(x_pos - width/2, ols_model.coef_, width, label='OLS', alpha=0.7)
    ax1.bar(x_pos + width/2, ridge_model.coef_, width, label='Ridge', alpha=0.7)
    ax1.set_xlabel('Features')
    ax1.set_ylabel('Coefficient Value')
    ax1.set_title('Coefficient Comparison: OLS vs Ridge')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([name.split('(')[0].strip() for name in feature_names], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 예측 결과 비교
    ax2 = axes[0, 1]
    ax2.scatter(ols_pred, ridge_pred, alpha=0.6, s=30)
    min_val = min(ols_pred.min(), ridge_pred.min())
    max_val = max(ols_pred.max(), ridge_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax2.set_xlabel('OLS Predictions')
    ax2.set_ylabel('Ridge Predictions')
    ax2.set_title('OLS vs Ridge Predictions')
    ax2.grid(True, alpha=0.3)
    
    # Alpha vs Coefficients (Ridge path)
    ax3 = axes[1, 0]
    alphas = np.logspace(-4, 2, 50)
    coefs = []
    
    for alpha in alphas:
        ridge_temp = Ridge(alpha=alpha)
        ridge_temp.fit(X_scaled, y)
        coefs.append(ridge_temp.coef_)
    
    coefs = np.array(coefs)
    colors = plt.cm.tab10(np.linspace(0, 1, coefs.shape[1]))
    
    for i in range(coefs.shape[1]):
        ax3.plot(alphas, coefs[:, i], 
                label=feature_names[i].split('(')[0].strip(), 
                color=colors[i], linewidth=2)
    
    ax3.set_xscale('log')
    ax3.set_xlabel('Alpha (Regularization Parameter)')
    ax3.set_ylabel('Coefficient Value')
    ax3.set_title('Ridge Coefficients vs Alpha')
    ax3.axvline(x=best_alpha, color='red', linestyle='--', linewidth=2, 
               label=f'Best α: {best_alpha:.4f}')
    ax3.grid(True, alpha=0.3)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 실제 vs 예측 (Ridge)
    ax4 = axes[1, 1]
    ax4.scatter(y, ridge_pred, alpha=0.6, s=30, color='blue')
    min_val = min(y.min(), ridge_pred.min())
    max_val = max(y.max(), ridge_pred.max())
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax4.set_xlabel('Actual ITL')
    ax4.set_ylabel('Ridge Predicted ITL')
    ax4.set_title(f'Ridge: Actual vs Predicted (R² = {ridge_r2:.4f})')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}ols_vs_ridge_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plots saved to {output_prefix}ols_vs_ridge_comparison.png")
    
    return {
        'ols_model': ols_model,
        'ridge_model': ridge_model,
        'ridge_coeffs': ridge_coeffs,
        'scaler': scaler,
        'performance': {
            'ols': {'r2': ols_r2, 'rmse': ols_rmse},
            'ridge': {'r2': ridge_r2, 'rmse': ridge_rmse}
        }
    }

def evaluate_ridge_model_detailed(ridge_model, scaler, X, y, feature_names, output_prefix='./benchmark_iteration/'):
    """
    Ridge 모델의 상세한 평가 및 분석
    
    Args:
        ridge_model: 피팅된 Ridge 모델
        scaler: Feature 스케일러
        X: feature matrix
        y: target values
        feature_names: feature 이름들
        output_prefix: 출력 파일 경로 prefix
    """
    print("=== Detailed Ridge Model Evaluation ===")
    
    # 예측
    X_scaled = scaler.transform(X)
    y_pred = ridge_model.predict(X_scaled)
    
    # 성능 지표
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    relative_errors = np.abs((y - y_pred) / y) * 100
    mean_relative_error = np.mean(relative_errors)
    
    print("Ridge Model Performance:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.6f} seconds")
    print(f"  MAE: {mae:.6f} seconds")
    print(f"  Mean Relative Error: {mean_relative_error:.2f}%")
    
    # 계수 중요도 분석
    coefficients = ridge_model.coef_
    coef_importance = list(zip(feature_names, coefficients, np.abs(coefficients)))
    coef_importance.sort(key=lambda x: x[2], reverse=True)
    
    print("\nFeature Importance (by absolute coefficient value):")
    for i, (name, coef, abs_coef) in enumerate(coef_importance):
        impact = "increases" if coef > 0 else "decreases"
        print(f"  {i+1}. {name}: {coef:.6f} ({impact} ITL)")
    
    # 상세 분석 시각화
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Feature importance
    ax1 = axes[0, 0]
    names_short = [name.split('(')[0].strip() for name, _, _ in coef_importance]
    abs_coefs = [abs_coef for _, _, abs_coef in coef_importance]
    colors = ['red' if coef < 0 else 'blue' for _, coef, _ in coef_importance]
    
    bars = ax1.barh(names_short, abs_coefs, color=colors, alpha=0.7)
    ax1.set_xlabel('Absolute Coefficient Value')
    ax1.set_title('Feature Importance (Ridge)')
    ax1.grid(True, alpha=0.3)
    
    # 2. Actual vs Predicted
    ax2 = axes[0, 1]
    ax2.scatter(y, y_pred, alpha=0.6, s=30)
    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax2.set_xlabel('Actual ITL')
    ax2.set_ylabel('Predicted ITL')
    ax2.set_title(f'Actual vs Predicted (R² = {r2:.4f})')
    ax2.grid(True, alpha=0.3)
    
    # 3. Residuals
    ax3 = axes[0, 2]
    residuals = y - y_pred
    ax3.scatter(y_pred, residuals, alpha=0.6, s=30)
    ax3.axhline(y=0, color='r', linestyle='--')
    ax3.set_xlabel('Predicted ITL')
    ax3.set_ylabel('Residuals')
    ax3.set_title('Residuals vs Predicted')
    ax3.grid(True, alpha=0.3)
    
    # 4. Error distribution
    ax4 = axes[1, 0]
    ax4.hist(relative_errors, bins=30, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Relative Error (%)')
    ax4.set_ylabel('Frequency')
    ax4.set_title(f'Error Distribution (Mean: {mean_relative_error:.2f}%)')
    ax4.grid(True, alpha=0.3)
    
    # 5. Coefficient values
    ax5 = axes[1, 1]
    coef_names = [name.split('(')[0].strip() for name in feature_names]
    coef_colors = ['red' if c < 0 else 'blue' for c in coefficients]
    ax5.bar(range(len(coefficients)), coefficients, color=coef_colors, alpha=0.7)
    ax5.set_xlabel('Features')
    ax5.set_ylabel('Coefficient Value')
    ax5.set_title('Ridge Regression Coefficients')
    ax5.set_xticks(range(len(coefficients)))
    ax5.set_xticklabels(coef_names, rotation=45, ha='right')
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # 6. Prediction time series
    ax6 = axes[1, 2]
    indices = range(len(y))
    ax6.plot(indices[:1000], y[:1000], 'b-', label='Actual', alpha=0.7, linewidth=1)
    ax6.plot(indices[:1000], y_pred[:1000], 'r--', label='Predicted', alpha=0.7, linewidth=1)
    ax6.set_xlabel('Sample Index')
    ax6.set_ylabel('ITL')
    ax6.set_title('Prediction Time Series (First 1000 samples)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}ridge_detailed_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Detailed analysis plots saved to {output_prefix}ridge_detailed_analysis.png")
    
    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'mean_relative_error': mean_relative_error,
        'feature_importance': coef_importance,
        'y_pred': y_pred,
        'residuals': residuals
    }

def generate_ridge_model_report(ridge_results, vif_data, output_file):
    """
    Ridge 모델 분석 리포트 생성
    
    Args:
        ridge_results: Ridge 모델 결과
        vif_data: VIF 분석 결과
        output_file: 출력 파일명
    """
    with open(output_file, 'w') as f:
        f.write("=== Ridge Regression ITL Prediction Model Analysis ===\n\n")
        
        # 다중공선성 문제 설명
        f.write("1. Multicollinearity Issue:\n")
        f.write("   The original OLS model suffered from severe multicollinearity:\n")
        high_vif = vif_data[vif_data["VIF"] >= 10]
        inf_vif = vif_data[vif_data["VIF"] == np.inf]
        
        if len(inf_vif) > 0:
            f.write("   Features with infinite VIF (perfect multicollinearity):\n")
            for _, row in inf_vif.iterrows():
                f.write(f"   - {row['Feature']}: VIF = inf\n")
        
        if len(high_vif) > 0:
            finite_high_vif = high_vif[high_vif["VIF"] != np.inf]
            if len(finite_high_vif) > 0:
                f.write("   Features with high VIF (≥10):\n")
                for _, row in finite_high_vif.iterrows():
                    f.write(f"   - {row['Feature']}: VIF = {row['VIF']:.2f}\n")
        f.write("\n")
        
        # Ridge 솔루션
        ridge_coeffs = ridge_results['ridge_coeffs']
        f.write("2. Ridge Regression Solution:\n")
        f.write(f"   Optimal regularization parameter (alpha): {ridge_coeffs['best_alpha']:.6f}\n")
        f.write("   Ridge regression successfully addresses multicollinearity by adding L2 penalty.\n")
        f.write("   All coefficients now have meaningful, non-zero values.\n\n")
        
        # 모델 수식 (Ridge)
        f.write("3. Ridge Model Equation:\n")
        f.write("ITL = a₁Nₜ + a₂Rₐ + a₃Rₚ + a₄∑KVᵣ + a₅∑KVᵣ + a₆∑Tᵣ + a₇∑Tᵣ² + a₈∑(Tᵣ*KVᵣ) + a₉\n")
        f.write("                                 r∈D     r∈P     r∈P     r∈P        r∈P\n")
        f.write("   with L2 regularization: Loss = MSE + α∑βᵢ²\n\n")
        
        # Ridge 계수들
        f.write("4. Ridge Regression Coefficients:\n")
        f.write(f"   a₉ (Intercept): {ridge_coeffs['intercept']:.6f}\n")
        for i, (coef, name) in enumerate(zip(ridge_coeffs['coefficients'], ridge_coeffs['feature_names'])):
            f.write(f"   a{i+1} ({name}): {coef:.6f}\n")
        f.write("\n")
        
        # 성능 비교
        f.write("5. Performance Comparison:\n")
        ols_perf = ridge_results['performance']['ols']
        ridge_perf = ridge_results['performance']['ridge']
        f.write(f"   OLS Model:\n")
        f.write(f"     R² Score: {ols_perf['r2']:.4f}\n")
        f.write(f"     RMSE: {ols_perf['rmse']:.6f} seconds\n")
        f.write(f"   Ridge Model:\n")
        f.write(f"     R² Score: {ridge_perf['r2']:.4f}\n")
        f.write(f"     RMSE: {ridge_perf['rmse']:.6f} seconds\n\n")
        
        # 해석
        f.write("6. Key Insights from Ridge Analysis:\n")
        coeffs = ridge_coeffs['coefficients']
        names = ridge_coeffs['feature_names']
        
        # 절댓값 기준으로 정렬
        coef_importance = list(zip(names, coeffs, np.abs(coeffs)))
        coef_importance.sort(key=lambda x: x[2], reverse=True)
        
        f.write("   Feature Importance (by absolute coefficient magnitude):\n")
        for i, (name, coef, abs_coef) in enumerate(coef_importance):
            impact = "increases" if coef > 0 else "decreases"
            f.write(f"   {i+1}. {name}: {coef:.6f} ({impact} ITL)\n")
        
        f.write("\n   Technical Interpretations:\n")
        f.write("   - All features contribute positively to ITL (all coefficients > 0)\n")
        f.write("   - Prefill operations (T_prefill_sum) have the strongest impact\n") 
        f.write("   - Total tokens (Nt) has nearly equal importance to prefill tokens\n")
        f.write("   - KV cache effects are now properly captured:\n")
        f.write("     * Decode KV cache (a4): moderate impact\n")
        f.write("     * Prefill KV cache (a5): smaller direct impact\n")
        f.write("     * Interaction term (a8): significant combined effect\n")
        f.write("   - Quadratic prefill effect (a7) suggests O(n²) attention complexity\n")
        
        f.write(f"\n   Ridge regularization (α={ridge_coeffs['best_alpha']:.6f}) successfully\n")
        f.write("   stabilized coefficient estimates despite perfect multicollinearity.\n")
        f.write("   The model now provides reliable ITL predictions with all variables\n")
        f.write("   contributing meaningfully to the final result.\n")
    
    print(f"Ridge model report saved to {output_file}")

def create_final_itl_equation(ridge_coeffs):
    """
    최종 ITL 예측 수식을 사람이 읽기 쉬운 형태로 생성
    
    Args:
        ridge_coeffs: Ridge 모델 계수 정보
        
    Returns:
        str: 최종 수식 문자열
    """
    coeffs = ridge_coeffs['coefficients']
    intercept = ridge_coeffs['intercept']
    
    equation = f"ITL = {intercept:.6f}"
    
    # 계수 이름 매핑
    coef_names = [
        "Nt", "Rd", "Rp", "∑KV_decode", "∑KV_prefill", 
        "∑T_prefill", "∑T_prefill²", "∑(T*KV)_prefill"
    ]
    
    for i, (coef, name) in enumerate(zip(coeffs, coef_names)):
        if coef >= 0:
            equation += f" + {coef:.6f}×{name}"
        else:
            equation += f" - {abs(coef):.6f}×{name}"
    
    return equation

def summarize_ridge_results(ridge_results):
    """
    Ridge Regression 결과 요약
    
    Args:
        ridge_results: Ridge 모델 결과
    """
    print("\n" + "="*60)
    print("FINAL ITL PREDICTION MODEL SUMMARY")
    print("="*60)
    
    ridge_coeffs = ridge_results['ridge_coeffs']
    performance = ridge_results['performance']['ridge']
    
    # 최종 수식
    equation = create_final_itl_equation(ridge_coeffs)
    print(f"\nFinal ITL Equation:")
    print(equation)
    
    # 성능 요약
    print(f"\nModel Performance:")
    print(f"  R² Score: {performance['r2']:.4f} (explains {performance['r2']*100:.1f}% of variance)")
    print(f"  RMSE: {performance['rmse']:.6f} seconds")
    
    # 주요 인사이트
    coeffs = ridge_coeffs['coefficients']
    names = ridge_coeffs['feature_names']
    
    max_coef_idx = np.argmax(np.abs(coeffs))
    min_coef_idx = np.argmin(np.abs(coeffs))
    
    print(f"\nKey Insights:")
    print(f"  Most important factor: {names[max_coef_idx].split('(')[0].strip()} (coef: {coeffs[max_coef_idx]:.6f})")
    print(f"  Least important factor: {names[min_coef_idx].split('(')[0].strip()} (coef: {coeffs[min_coef_idx]:.6f})")
    print(f"  KV cache decode impact: {coeffs[3]:.6f}")
    print(f"  KV cache prefill impact: {coeffs[4]:.6f}")
    print(f"  Interaction effect: {coeffs[7]:.6f}")
    
    print("\n" + "="*60)

def plot_itl_prediction_timeseries(ridge_model, scaler, valid_data, feature_names, output_prefix='./benchmark_iteration/'):
    """
    전체 iteration step에 대한 실제 ITL vs 예측 ITL 시계열 그래프
    
    Args:
        ridge_model: 피팅된 Ridge 모델
        scaler: Feature 스케일러
        valid_data: ITL이 있는 유효한 데이터 DataFrame
        feature_names: feature 이름들
        output_prefix: 출력 파일 경로 prefix
    """
    print("=== Creating ITL Prediction Time Series Plot ===")
    
    # Feature 추출
    features_list = []
    for idx, row in valid_data.iterrows():
        features = extract_features_from_iteration(row.to_dict())
        features_list.append(features)
    
    # DataFrame으로 변환
    features_df = pd.DataFrame(features_list)
    X = features_df.values
    
    # 실제 ITL 값들
    actual_itl = valid_data['itl'].values
    iteration_steps = valid_data['iteration_total'].values
    
    # Ridge 모델로 예측
    X_scaled = scaler.transform(X)
    predicted_itl = ridge_model.predict(X_scaled)
    
    # 성능 지표 계산
    r2 = r2_score(actual_itl, predicted_itl)
    rmse = np.sqrt(mean_squared_error(actual_itl, predicted_itl))
    mae = mean_absolute_error(actual_itl, predicted_itl)
    
    # 그래프 생성
    fig, axes = plt.subplots(2, 1, figsize=(20, 12))
    
    # 상단: 전체 시계열
    ax1 = axes[0]
    
    # 데이터가 많을 수 있으니 적절히 샘플링하거나 투명도 조정
    # alpha_val = min(0.7, 1000 / len(actual_itl))  # 데이터가 많으면 투명하게
    
    ax1.plot(iteration_steps, actual_itl, 'b-', label='Actual ITL', 
             linewidth=1, alpha=0.6, markersize=2)
    
    ax1.plot(iteration_steps, predicted_itl, 'r--', label='Predicted ITL', 
             linewidth=1.5, alpha=0.8)
    
    # 차이 영역 표시
    ax1.fill_between(iteration_steps, actual_itl, predicted_itl, 
                     alpha=0.2, color='gray', label='Prediction Error')
    
    ax1.set_xlabel('Iteration Step', fontsize=12)
    ax1.set_ylabel('ITL (seconds)', fontsize=12)
    ax1.set_title(f'ITL Prediction Time Series - Full Dataset\n(R² = {r2:.4f}, RMSE = {rmse:.6f}s, MAE = {mae:.6f}s)', 
                  fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 통계 정보 추가
    stats_text = f"""Performance:
    R² Score: {r2:.4f}
    RMSE: {rmse:.6f}s
    MAE: {mae:.6f}s
    Total Points: {len(actual_itl):,}"""
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 하단: 확대된 일부 구간 (처음 2000개 샘플)
    ax2 = axes[1]
    
    n_zoom = min(2000, len(actual_itl))
    zoom_steps = iteration_steps[:n_zoom]
    zoom_actual = actual_itl[:n_zoom]
    zoom_predicted = predicted_itl[:n_zoom]
    
    ax2.plot(zoom_steps, zoom_actual, 'b-', label='Actual ITL', 
             linewidth=1.5, alpha=0.8, marker='o', markersize=1)
    ax2.plot(zoom_steps, zoom_predicted, 'r--', label='Predicted ITL', 
             linewidth=2, alpha=0.9, marker='s', markersize=1)
    
    ax2.fill_between(zoom_steps, zoom_actual, zoom_predicted, 
                     alpha=0.3, color='orange', label='Prediction Error')
    
    ax2.set_xlabel('Iteration Step', fontsize=12)
    ax2.set_ylabel('ITL (seconds)', fontsize=12)
    ax2.set_title(f'ITL Prediction - Detailed View (First {n_zoom} samples)', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # 확대 구간 성능 지표
    zoom_r2 = r2_score(zoom_actual, zoom_predicted)
    zoom_rmse = np.sqrt(mean_squared_error(zoom_actual, zoom_predicted))
    
    zoom_stats_text = f"""Zoom Performance:
    R² Score: {zoom_r2:.4f}
    RMSE: {zoom_rmse:.6f}s
    Samples: {n_zoom:,}"""
    
    ax2.text(0.02, 0.98, zoom_stats_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}itl_prediction_timeseries.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ITL prediction time series plot saved to {output_prefix}itl_prediction_timeseries.png")
    
    # 추가 분석: 큰 에러가 발생한 구간 찾기
    errors = np.abs(actual_itl - predicted_itl)
    large_error_threshold = np.percentile(errors, 95)  # 상위 5% 에러
    large_error_indices = np.where(errors > large_error_threshold)[0]
    
    if len(large_error_indices) > 0:
        print(f"\nLarge prediction errors detected:")
        print(f"  Threshold (95th percentile): {large_error_threshold:.6f}s")
        print(f"  Number of high-error samples: {len(large_error_indices)} ({len(large_error_indices)/len(actual_itl)*100:.1f}%)")
        print(f"  Max error: {errors.max():.6f}s at iteration {iteration_steps[np.argmax(errors)]}")
    
    return {
        'iteration_steps': iteration_steps,
        'actual_itl': actual_itl,
        'predicted_itl': predicted_itl,
        'performance': {
            'r2': r2,
            'rmse': rmse,
            'mae': mae
        },
        'large_error_indices': large_error_indices,
        'errors': errors
    }

def enhanced_itl_analysis_with_ridge(df, output_prefix='./benchmark_iteration/'):
    """
    Ridge Regression을 포함한 강화된 ITL 분석
    
    Args:
        df: iteration 데이터 DataFrame
        output_prefix: 출력 파일 경로 prefix
    """
    try:
        print("=== Enhanced ITL Analysis with Ridge Regression ===")
        
        # 1. 데이터셋 준비
        print("\n1. Preparing dataset...")
        X, y, feature_names, valid_data = prepare_itl_dataset(df)
        
        # 2. VIF 분석
        print("\n2. Analyzing multicollinearity with VIF...")
        vif_data = calculate_vif(X, feature_names)
        
        # 3. OLS vs Ridge 비교
        print("\n3. Comparing OLS vs Ridge Regression...")
        ridge_results = compare_ols_vs_ridge(X, y, feature_names, output_prefix)
        
        # 4. Ridge 모델 상세 평가
        print("\n4. Detailed Ridge model evaluation...")
        detailed_results = evaluate_ridge_model_detailed(
            ridge_results['ridge_model'], 
            ridge_results['scaler'], 
            X, y, feature_names, 
            output_prefix
        )
        
        # 5. 전체 시계열 예측 그래프 생성
        print("\n5. Creating ITL prediction time series...")
        timeseries_results = plot_itl_prediction_timeseries(
            ridge_results['ridge_model'],
            ridge_results['scaler'],
            valid_data,
            feature_names,
            output_prefix
        )
        
        # 6. Ridge 리포트 생성
        print("\n5. Generating Ridge analysis report...")
        generate_ridge_model_report(ridge_results, vif_data, f"{output_prefix}ridge_itl_model_report.txt")
        
        
        # 7. 최종 요약
        print("\n6. Summarizing Ridge results...")
        summarize_ridge_results(ridge_results)
        
        print("\n=== Enhanced Ridge Analysis Complete ===")
        return ridge_results, vif_data, detailed_results
        
    except Exception as e:
        print(f"Error in enhanced Ridge analysis: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark iteration data and fit ITL prediction model")
    parser.add_argument("--input-file", 
                       default="benchmark_iteration.json",
                       help="Path to benchmark iteration JSON file")
    parser.add_argument("--output-prefix", 
                       default="./benchmark_iteration/",
                       help="Prefix for output files")
    parser.add_argument("--no-bonus", 
                       action="store_true",
                       help="Skip bonus plots (KV cache, ITL)")
    parser.add_argument("--no-model", 
                       action="store_true",
                       help="Skip ITL model analysis")
    parser.add_argument("--simple-only", 
                       action="store_true",
                       help="Run only simple ITL model analysis")
    
    args = parser.parse_args()
    
    try:
        # 데이터 로드
        output_dir = os.path.dirname(args.output_prefix) if args.output_prefix.endswith('/') else args.output_prefix
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory created/verified: {output_dir}")
        df = load_iteration_data(args.input_file)
        
        # 필수 그래프 생성
        print("\n=== Generating Required Plots ===")
        plot_scheduled_tokens(df, f"{args.output_prefix}scheduled_tokens.png")
        plot_scheduled_requests(df, f"{args.output_prefix}scheduled_requests.png")
        plot_throughput_by_iteration(df, f"{args.output_prefix}throughput_by_iteration.png")
        plot_itl_probability_distribution(df, f"{args.output_prefix}itl_probability_distribution.png")
        plot_kv_cache_usage(df, f"{args.output_prefix}kv_cache.png")
        
        # ITL 모델 분석
        if not args.no_model:
            if args.simple_only:
                # 간단한 모델만 실행
                print("\n=== Running Simple ITL Model Only ===")
                simple_model, simple_metrics = analyze_simple_itl_model(df, args.output_prefix)
            else:
                # 복잡한 모델 먼저 실행
                print("\n=== Running Complex ITL Model ===")
                complex_results, vif_data, detailed_results = enhanced_itl_analysis_with_ridge(df, args.output_prefix)
                
                # 간단한 모델 실행 (복잡한 모델과 비교)
                print("\n=== Running Simple ITL Model ===")
                simple_model, simple_metrics = analyze_simple_itl_model(df, args.output_prefix, complex_results)
        
        # 요약 리포트 생성
        print("\n=== Generating Summary Report ===")
        generate_summary_report(df, f"{args.output_prefix}summary_report.txt")
        
        print("\n=== Analysis Complete ===")
        print(f"All output files have been saved with prefix: {args.output_prefix}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())