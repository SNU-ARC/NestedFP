import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse

def load_benchmark_data(json_file):
    """benchmark_slo.json 파일 로드"""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        print(f"Successfully loaded data from {json_file}")
        return data
    except FileNotFoundError:
        print(f"Error: File {json_file} not found")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_file}")
        return None

def calculate_detailed_statistics(results):
    """상세한 통계 계산"""
    if not results:
        print("No results to analyze")
        return {}
    
    # 성공한 요청만 필터링
    successful_results = [r for r in results if r['success']]
    total_requests = len(results)
    successful_requests = len(successful_results)
    
    if not successful_results:
        print("No successful requests found")
        return {}
    
    # 데이터 추출
    ttft_values = [r['ttft'] for r in successful_results if r['ttft'] > 0]
    tpot_values = [r['tpot'] for r in successful_results if r['tpot'] > 0]
    latency_values = [r['latency'] for r in successful_results]
    context_tokens = [r['context_tokens'] for r in successful_results]
    generated_tokens = [r['generated_tokens'] for r in successful_results]
    actual_generated_tokens = [r['actual_generated_tokens'] for r in successful_results]
    
    def calculate_stats(values, name):
        """통계값 계산 헬퍼 함수"""
        if not values:
            return {}
        
        return {
            'count': len(values),
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'p10': np.percentile(values, 10),
            'p25': np.percentile(values, 25),
            'p75': np.percentile(values, 75),
            'p90': np.percentile(values, 90),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99),
        }
    
    # 전체 통계
    stats = {
        'summary': {
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'failed_requests': total_requests - successful_requests,
            'success_rate': successful_requests / total_requests if total_requests > 0 else 0,
        },
        'ttft': calculate_stats(ttft_values, 'TTFT'),
        'tpot': calculate_stats(tpot_values, 'TPOT'),
        'latency': calculate_stats(latency_values, 'Latency'),
        'context_tokens': calculate_stats(context_tokens, 'Context Tokens'),
        'generated_tokens': calculate_stats(generated_tokens, 'Generated Tokens (Target)'),
        'actual_generated_tokens': calculate_stats(actual_generated_tokens, 'Actual Generated Tokens'),
    }
    
    # SLO 제안값 계산
    if ttft_values and tpot_values:
        stats['slo_suggestions'] = {}
        multipliers = [5, 10, 15, 20, 25]
        
        for mult in multipliers:
            stats['slo_suggestions'][f'{mult}x'] = {
                'ttft_slo': np.mean(ttft_values) * mult,
                'tpot_slo': np.mean(tpot_values) * mult,
            }
    
    return stats

def save_statistics_to_file(stats, experiment_info, output_dir):
    """통계를 텍스트 파일로 저장"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats_file = os.path.join(output_dir, f"slo_statistics_{timestamp}.txt")
    
    with open(stats_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("SLO BENCHMARK ANALYSIS REPORT\n")
        f.write("="*60 + "\n\n")
        
        # 실험 정보
        f.write("EXPERIMENT INFORMATION:\n")
        f.write("-"*30 + "\n")
        for key, value in experiment_info.items():
            f.write(f"{key}: {value}\n")
        f.write(f"Analysis generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 요약 통계
        f.write("SUMMARY STATISTICS:\n")
        f.write("-"*30 + "\n")
        summary = stats['summary']
        f.write(f"Total Requests: {summary['total_requests']}\n")
        f.write(f"Successful Requests: {summary['successful_requests']}\n")
        f.write(f"Failed Requests: {summary['failed_requests']}\n")
        f.write(f"Success Rate: {summary['success_rate']:.2%}\n\n")
        
        # 각 메트릭별 상세 통계
        metrics = ['ttft', 'tpot', 'latency', 'context_tokens', 'generated_tokens', 'actual_generated_tokens']
        metric_names = ['TTFT (seconds)', 'TPOT (seconds)', 'Latency (seconds)', 
                       'Context Tokens', 'Generated Tokens (Target)', 'Actual Generated Tokens']
        
        for metric, name in zip(metrics, metric_names):
            if metric in stats and stats[metric]:
                f.write(f"{name.upper()}:\n")
                f.write("-"*30 + "\n")
                s = stats[metric]
                f.write(f"Count: {s['count']}\n")
                f.write(f"Mean: {s['mean']:.6f}\n")
                f.write(f"Median: {s['median']:.6f}\n")
                f.write(f"Std Dev: {s['std']:.6f}\n")
                f.write(f"Min: {s['min']:.6f}\n")
                f.write(f"Max: {s['max']:.6f}\n")
                f.write(f"P10: {s['p10']:.6f}\n")
                f.write(f"P25: {s['p25']:.6f}\n")
                f.write(f"P75: {s['p75']:.6f}\n")
                f.write(f"P90: {s['p90']:.6f}\n")
                f.write(f"P95: {s['p95']:.6f}\n")
                f.write(f"P99: {s['p99']:.6f}\n\n")
        
        # SLO 제안값
        if 'slo_suggestions' in stats:
            f.write("SLO SUGGESTIONS:\n")
            f.write("-"*30 + "\n")
            for multiplier, slo_values in stats['slo_suggestions'].items():
                f.write(f"{multiplier} Multiplier:\n")
                f.write(f"  TTFT SLO: {slo_values['ttft_slo']:.6f} seconds\n")
                f.write(f"  TPOT SLO: {slo_values['tpot_slo']:.6f} seconds\n")
            f.write("\n")
    
    print(f"Statistics saved to: {stats_file}")
    return stats_file

def create_performance_plots(results, output_dir):
    """성능 관련 플롯 생성"""
    if not results:
        print("No results to plot")
        return
    
    # 성공한 요청만 필터링
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        print("No successful results to plot")
        return
    
    # 데이터 준비
    df = pd.DataFrame(successful_results)
    df = df[df['ttft'] > 0]  # 유효한 TTFT 값만
    
    # 스타일 설정
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. TTFT 히스토그램
    plt.figure(figsize=(10, 6))
    plt.hist(df['ttft'], bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('TTFT (seconds)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Time to First Token (TTFT)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ttft_histogram.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. TPOT 히스토그램
    tpot_data = df[df['tpot'] > 0]['tpot']
    if len(tpot_data) > 0:
        plt.figure(figsize=(10, 6))
        plt.hist(tpot_data, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('TPOT (seconds)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Time per Output Token (TPOT)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'tpot_histogram.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. TTFT vs TPOT 산점도
    tpot_valid = df[df['tpot'] > 0]
    if len(tpot_valid) > 0:
        plt.figure(figsize=(10, 8))
        plt.scatter(tpot_valid['tpot'], tpot_valid['ttft'], alpha=0.6, s=50)
        plt.xlabel('TPOT (seconds)')
        plt.ylabel('TTFT (seconds)')
        plt.title('TTFT vs TPOT Scatter Plot')
        plt.grid(True, alpha=0.3)
        
        # 평균선 추가
        avg_tpot = tpot_valid['tpot'].mean()
        avg_ttft = tpot_valid['ttft'].mean()
        plt.axvline(avg_tpot, color='red', linestyle='--', alpha=0.8, label=f'Avg TPOT: {avg_tpot:.4f}s')
        plt.axhline(avg_ttft, color='green', linestyle='--', alpha=0.8, label=f'Avg TTFT: {avg_ttft:.4f}s')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ttft_vs_tpot_scatter.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Context Tokens vs TTFT
    plt.figure(figsize=(10, 8))
    plt.scatter(df['context_tokens'], df['ttft'], alpha=0.6, s=50)
    plt.xlabel('Context Tokens')
    plt.ylabel('TTFT (seconds)')
    plt.title('Context Tokens vs TTFT')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'context_tokens_vs_ttft.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Generated Tokens vs TPOT
    if len(tpot_valid) > 0:
        plt.figure(figsize=(10, 8))
        plt.scatter(tpot_valid['actual_generated_tokens'], tpot_valid['tpot'], alpha=0.6, s=50)
        plt.xlabel('Actual Generated Tokens')
        plt.ylabel('TPOT (seconds)')
        plt.title('Generated Tokens vs TPOT')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'generated_tokens_vs_tpot.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 6. Request 순서별 TTFT/TPOT 시계열
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(df['request_id'], df['ttft'], 'o-', markersize=3, linewidth=1, alpha=0.7)
    plt.xlabel('Request ID')
    plt.ylabel('TTFT (seconds)')
    plt.title('TTFT over Request Sequence')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    if len(tpot_valid) > 0:
        plt.plot(tpot_valid['request_id'], tpot_valid['tpot'], 'o-', markersize=3, linewidth=1, alpha=0.7, color='orange')
        plt.xlabel('Request ID')
        plt.ylabel('TPOT (seconds)')
        plt.title('TPOT over Request Sequence')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_time_series.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Performance plots generated:")
    print("  - ttft_histogram.png")
    print("  - tpot_histogram.png")
    print("  - ttft_vs_tpot_scatter.png")
    print("  - context_tokens_vs_ttft.png")
    print("  - generated_tokens_vs_tpot.png")
    print("  - performance_time_series.png")

def create_token_distribution_plots(results, output_dir):
    """토큰 분포 관련 플롯 생성"""
    if not results:
        return
    
    successful_results = [r for r in results if r['success']]
    if not successful_results:
        return
    
    df = pd.DataFrame(successful_results)
    
    # 1. Context Tokens 분포
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(df['context_tokens'], bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Context Tokens')
    plt.ylabel('Frequency')
    plt.title('Distribution of Context Tokens')
    plt.grid(True, alpha=0.3)
    
    # 2. Generated Tokens 분포 (Target vs Actual)
    plt.subplot(1, 2, 2)
    plt.hist(df['generated_tokens'], bins=30, alpha=0.7, label='Target', edgecolor='black')
    plt.hist(df['actual_generated_tokens'], bins=30, alpha=0.7, label='Actual', edgecolor='black')
    plt.xlabel('Generated Tokens')
    plt.ylabel('Frequency')
    plt.title('Distribution of Generated Tokens')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'token_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Target vs Actual Generated Tokens 산점도
    plt.figure(figsize=(10, 8))
    plt.scatter(df['generated_tokens'], df['actual_generated_tokens'], alpha=0.6, s=50)
    plt.xlabel('Target Generated Tokens')
    plt.ylabel('Actual Generated Tokens')
    plt.title('Target vs Actual Generated Tokens')
    
    # 대각선 (perfect match) 추가
    max_val = max(df['generated_tokens'].max(), df['actual_generated_tokens'].max())
    plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.8, label='Perfect Match')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'target_vs_actual_tokens.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Token distribution plots generated:")
    print("  - token_distributions.png")
    print("  - target_vs_actual_tokens.png")

def main():
    parser = argparse.ArgumentParser(description="Analyze SLO benchmark results")
    parser.add_argument("--json-file", 
                       default="benchmark_slo.json",
                       help="Path to benchmark_slo.json file")
    parser.add_argument("--output-dir", 
                       default="./benchmark_slo",
                       help="Output directory for analysis results")
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # 데이터 로드
    data = load_benchmark_data(args.json_file)
    if data is None:
        return
    
    # 실험 정보 및 결과 추출
    experiment_info = data.get('experiment_info', {})
    results = data.get('results', [])
    
    print(f"\nAnalyzing {len(results)} requests...")
    
    # 통계 계산
    stats = calculate_detailed_statistics(results)
    
    if not stats:
        print("No statistics to analyze")
        return
    
    # 통계 출력
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    summary = stats['summary']
    print(f"Total Requests: {summary['total_requests']}")
    print(f"Successful Requests: {summary['successful_requests']}")
    print(f"Success Rate: {summary['success_rate']:.2%}")
    
    if 'ttft' in stats and stats['ttft']:
        print(f"\nTTFT: Mean={stats['ttft']['mean']:.4f}s, P90={stats['ttft']['p90']:.4f}s, P99={stats['ttft']['p99']:.4f}s")
    
    if 'tpot' in stats and stats['tpot']:
        print(f"TPOT: Mean={stats['tpot']['mean']:.4f}s, P90={stats['tpot']['p90']:.4f}s, P99={stats['tpot']['p99']:.4f}s")
    
    if 'context_tokens' in stats and stats['context_tokens']:
        print(f"Context Tokens: Mean={stats['context_tokens']['mean']:.1f}, Range={stats['context_tokens']['min']:.0f}-{stats['context_tokens']['max']:.0f}")
    
    if 'actual_generated_tokens' in stats and stats['actual_generated_tokens']:
        print(f"Generated Tokens: Mean={stats['actual_generated_tokens']['mean']:.1f}, Range={stats['actual_generated_tokens']['min']:.0f}-{stats['actual_generated_tokens']['max']:.0f}")
    
    # SLO 제안값 출력
    if 'slo_suggestions' in stats:
        print(f"\nSLO SUGGESTIONS:")
        for multiplier, slo_values in stats['slo_suggestions'].items():
            print(f"  {multiplier}: TTFT={slo_values['ttft_slo']:.4f}s, TPOT={slo_values['tpot_slo']:.4f}s")
    
    # 파일로 저장
    stats_file = save_statistics_to_file(stats, experiment_info, args.output_dir)
    
    # 플롯 생성
    print(f"\nGenerating plots...")
    create_performance_plots(results, args.output_dir)
    create_token_distribution_plots(results, args.output_dir)
    
    print(f"\nAnalysis complete! Results saved in: {args.output_dir}")

if __name__ == "__main__":
    main()