import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from typing import Dict, List, Any

def load_benchmark_request(json_file: str) -> List[Dict[str, Any]]:
    """JSON 파일에서 benchmark 결과를 로드"""
    with open(json_file, 'r') as f:
        results = json.load(f)
    print(f"Loaded {len(results)} benchmark requests from {json_file}")
    return results

def reconstruct_global_data(results: List[Dict[str, Any]]) -> tuple:
    """benchmark 결과로부터 전역 변수들을 복원"""
    ttft_graph = {'iteration_step': [], 'ttft': []}
    iter_tpot_graph = {}
    iter_num_prefill_graph = {}
    iter_num_decode_graph = {}
    
    for result in results:
        if not result['success']:
            continue
            
        # TTFT 데이터 복원
        if result['ttft'] > 0 and result.get('iteration_data'):
            # 첫 번째 토큰의 iteration_total을 찾기
            first_iteration = result['iteration_data'][0]['iteration_total']
            ttft_graph['iteration_step'].append(first_iteration)
            ttft_graph['ttft'].append(result['ttft'])
        
        # iteration 데이터에서 각종 메트릭 복원
        iteration_data = result.get('iteration_data', [])
        token_arrival_times = result.get('token_arrival_times', [])
        
        for i, data in enumerate(iteration_data):
            iteration_total = data['iteration_total']
            timestamp = data['timestamp']
            # Token latency 계산 (첫 번째 토큰이 아닌 경우)
            if i > 0 and i < len(token_arrival_times):
                prev_timestamp = token_arrival_times[i-1]
                token_latency = timestamp - prev_timestamp
                if iteration_total not in iter_tpot_graph:
                    iter_tpot_graph[iteration_total] = []
                iter_tpot_graph[iteration_total].append(token_latency)
        
        # ITL 데이터에서 token latency 복원 (더 정확한 방법)
        itl_data = result.get('itl', [])
        if itl_data and iteration_data:
            for i, latency in enumerate(itl_data):
                # i+1은 두 번째 토큰부터 시작 (첫 번째는 TTFT)
                if i+1 < len(iteration_data):
                    iteration_total = iteration_data[i+1]['iteration_total']
                    if iteration_total not in iter_tpot_graph:
                        iter_tpot_graph[iteration_total] = []
                    iter_tpot_graph[iteration_total].append(latency)
    
    print(f"Reconstructed data:")
    print(f"  TTFT points: {len(ttft_graph['ttft'])}")
    print(f"  TPOT iteration steps: {len(iter_tpot_graph)}")
    
    return ttft_graph, iter_tpot_graph, iter_num_prefill_graph, iter_num_decode_graph

def save_ttft_plot(dic, x, y, file_name, output_dir):
    """TTFT 그래프 저장"""
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.DataFrame(dic)
    if len(df) > 0:
        df = df.sort_values(x)
        plt.figure(figsize=(10, 6))
        plt.plot(df[x], df[y], '-', linewidth=1, alpha=0.6, color='lightblue')
        plt.scatter(df[x], df[y], s=50, alpha=0.9, color='red', edgecolors='darkred', linewidth=1)
        plt.xlabel('First Token Iteration Step')
        plt.ylabel('TTFT [s]')
        plt.title('TTFT by First Token Iteration Step')
        plt.grid(True, alpha=0.3)
        
        output_path = os.path.join(output_dir, f'{file_name}.png')
        plt.savefig(output_path, dpi=1000)
        plt.close()
        print(f"TTFT plot saved to {output_path}")

def save_tpot_plots(iter_dict, file_name, output_dir):
    """서버 iteration step별 token latency 그래프 저장"""
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    if not iter_dict:
        print("No TPOT data to plot")
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
    
    plt.figure(figsize=(12, 6))
    plt.plot(iterations, latencies, 'o-', markersize=4, linewidth=1, alpha=0.7, color='blue')
    plt.xlabel('Server Iteration Step')
    plt.ylabel('Token Latency [s]')
    plt.title('Token Latency by Iteration Step')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{file_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"TPOT plot saved to {output_path}")
    
    # 통계 정보 출력
    print(f"\nTPOT Statistics:")
    print(f"Total iteration steps executed: {len(iter_dict)}")
    print(f"Unique iteration step range: {min(iterations)} - {max(iterations)}")
    print(f"Total tokens processed: {len(latencies)}")
    if latencies:
        print(f"Average token latency: {np.mean(latencies):.4f}s")
        print(f"Min token latency: {min(latencies):.4f}s")
        print(f"Max token latency: {max(latencies):.4f}s")
        


def create_performance_scatter_plots(results, output_dir, middle_ratio=0.8, slo_tpot=None, slo_ttft=None):
    """TTFT vs TPOT 산포도 생성"""
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    successful_results = [r for r in results if r['success']]
    successful_results.sort(key=lambda x: x['request_id'])
    
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
    
    # 통계 계산
    latency_values = [r['latency'] for r in filtered_results]
    ttft_values = [r['ttft'] for r in filtered_results if r['ttft'] > 0]
    tpot_values = [r['tpot'] for r in filtered_results if r['tpot'] > 0]
    
    stats = {}
    stats['filtered_count'] = len(filtered_results)
    stats['filter_range'] = f"{start_idx}-{end_idx-1}"
    stats['avg_latency'] = np.mean(latency_values)
    
    if ttft_values:
        stats['p50_ttft'] = np.percentile(ttft_values, 50)
        stats['p90_ttft'] = np.percentile(ttft_values, 90)
        stats['p99_ttft'] = np.percentile(ttft_values, 99)
    
    if tpot_values:
        stats['p50_tpot'] = np.percentile(tpot_values, 50)
        stats['p90_tpot'] = np.percentile(tpot_values, 90)
        stats['p99_tpot'] = np.percentile(tpot_values, 99)
    
    # 통계 출력
    print(f"\n--- Performance Statistics (Middle {middle_ratio*100}% of requests) ---")
    print(f"Filtered data range: requests {start_idx} to {end_idx-1} ({len(filtered_results)} requests)")
    print(f"Average latency: {stats.get('avg_latency', 0):.4f}s")
    if 'p50_ttft' in stats:
        print(f"P50 TTFT: {stats['p50_ttft']:.4f}s")
        print(f"P90 TTFT: {stats['p90_ttft']:.4f}s")
        print(f"P99 TTFT: {stats['p99_ttft']:.4f}s")
    if 'p50_tpot' in stats:
        print(f"P50 TPOT: {stats['p50_tpot']:.4f}s")
        print(f"P90 TPOT: {stats['p90_tpot']:.4f}s")
        print(f"P99 TPOT: {stats['p99_tpot']:.4f}s")
    
    # 산포도 생성
    paired_data = [(r['tpot'], r['ttft']) for r in filtered_results 
                   if r['ttft'] > 0 and r['tpot'] > 0]
    
    if not paired_data:
        print("No paired TTFT/TPOT data to plot")
        return stats, success_count
    
    paired_tpot, paired_ttft = zip(*paired_data)
    
    # SLO 만족 비율 계산 (tight, loosed 각각)
    if slo_tpot is not None and len(slo_tpot) == 2:
        slo_tpot_tight, slo_tpot_loosed = slo_tpot
        tpot_tight_satisfied = sum(1 for tpot in paired_tpot if tpot <= slo_tpot_tight)
        tpot_loosed_satisfied = sum(1 for tpot in paired_tpot if tpot <= slo_tpot_loosed)
        
        total_paired = len(paired_data)
        tpot_tight_ratio = tpot_tight_satisfied / total_paired * 100
        tpot_loosed_ratio = tpot_loosed_satisfied / total_paired * 100
        
        print(f"\n--- TPOT SLO Satisfaction ---")
        print(f"TPOT Tight SLO ({slo_tpot_tight:.3f}s) satisfied: {tpot_tight_satisfied}/{total_paired} ({tpot_tight_ratio:.1f}%)")
        print(f"TPOT Loosed SLO ({slo_tpot_loosed:.3f}s) satisfied: {tpot_loosed_satisfied}/{total_paired} ({tpot_loosed_ratio:.1f}%)")
    
    if slo_ttft is not None and len(slo_ttft) == 2:
        slo_ttft_tight, slo_ttft_loosed = slo_ttft
        ttft_tight_satisfied = sum(1 for ttft in paired_ttft if ttft <= slo_ttft_tight)
        ttft_loosed_satisfied = sum(1 for ttft in paired_ttft if ttft <= slo_ttft_loosed)
        
        total_paired = len(paired_data)
        ttft_tight_ratio = ttft_tight_satisfied / total_paired * 100
        ttft_loosed_ratio = ttft_loosed_satisfied / total_paired * 100
        
        print(f"\n--- TTFT SLO Satisfaction ---")
        print(f"TTFT Tight SLO ({slo_ttft_tight:.3f}s) satisfied: {ttft_tight_satisfied}/{total_paired} ({ttft_tight_ratio:.1f}%)")
        print(f"TTFT Loosed SLO ({slo_ttft_loosed:.3f}s) satisfied: {ttft_loosed_satisfied}/{total_paired} ({ttft_loosed_ratio:.1f}%)")
    
    # Both SLO 만족 비율 계산
    if (slo_tpot is not None and len(slo_tpot) == 2 and 
        slo_ttft is not None and len(slo_ttft) == 2):
        both_tight_satisfied = sum(1 for tpot, ttft in paired_data 
                                 if tpot <= slo_tpot_tight and ttft <= slo_ttft_tight)
        both_loosed_satisfied = sum(1 for tpot, ttft in paired_data 
                                  if tpot <= slo_tpot_loosed and ttft <= slo_ttft_loosed)
        
        both_tight_ratio = both_tight_satisfied / total_paired * 100
        both_loosed_ratio = both_loosed_satisfied / total_paired * 100
        
        print(f"\n--- Combined SLO Satisfaction ---")
        print(f"Both Tight SLOs satisfied: {both_tight_satisfied}/{total_paired} ({both_tight_ratio:.1f}%)")
        print(f"Both Loosed SLOs satisfied: {both_loosed_satisfied}/{total_paired} ({both_loosed_ratio:.1f}%)")
    
    plt.figure(figsize=(12, 8))
    plt.scatter(paired_tpot, paired_ttft, alpha=0.7, s=50, color='blue', edgecolors='darkblue', linewidth=0.5)
    plt.xlabel('TPOT [s]')
    plt.ylabel('TTFT [s]')
    plt.title(f'TTFT vs TPOT Scatter Plot (Middle {middle_ratio*100}% of requests)')
    plt.grid(True, alpha=0.3)
    
    # P50과 P90 값 계산
    p50_tpot = np.percentile(paired_tpot, 50)
    p50_ttft = np.percentile(paired_ttft, 50)
    p90_tpot = np.percentile(paired_tpot, 90)
    p90_ttft = np.percentile(paired_ttft, 90)
    
    # P50과 P90 선 추가 (연한 색상의 점선)
    plt.axvline(p50_tpot, color='lightcoral', linestyle=':', alpha=0.8, linewidth=2, 
                label=f'P50 TPOT: {p50_tpot:.3f}s')
    plt.axhline(p50_ttft, color='lightblue', linestyle=':', alpha=0.8, linewidth=2, 
                label=f'P50 TTFT: {p50_ttft:.3f}s')
    plt.axvline(p90_tpot, color='lightcoral', linestyle='--', alpha=0.8, linewidth=2, 
                label=f'P90 TPOT: {p90_tpot:.3f}s')
    plt.axhline(p90_ttft, color='lightblue', linestyle='--', alpha=0.8, linewidth=2, 
                label=f'P90 TTFT: {p90_ttft:.3f}s')
    
    # SLO 라인 추가 (tight, loosed 각각 실선으로)
    if slo_tpot is not None and len(slo_tpot) == 2:
        slo_tpot_tight, slo_tpot_loosed = slo_tpot
        plt.axvline(slo_tpot_tight, color='red', linestyle='-', alpha=0.9, linewidth=3, 
                    label=f'Tight TPOT SLO: {slo_tpot_tight:.3f}s')
        plt.axvline(slo_tpot_loosed, color='red', linestyle='-', alpha=0.6, linewidth=3, 
                    label=f'Loosed TPOT SLO: {slo_tpot_loosed:.3f}s')
    
    if slo_ttft is not None and len(slo_ttft) == 2:
        slo_ttft_tight, slo_ttft_loosed = slo_ttft
        plt.axhline(slo_ttft_tight, color='blue', linestyle='-', alpha=0.9, linewidth=3, 
                    label=f'Tight TTFT SLO: {slo_ttft_tight:.3f}s')
        plt.axhline(slo_ttft_loosed, color='blue', linestyle='-', alpha=0.6, linewidth=3, 
                    label=f'Loosed TTFT SLO: {slo_ttft_loosed:.3f}s')
    
    # 축 범위 고정 (loosed SLO 기준으로)
    if slo_tpot is not None and len(slo_tpot) == 2:
        plt.xlim(0, slo_tpot[1] * 2.5)  # loosed SLO 기준
    if slo_ttft is not None and len(slo_ttft) == 2:
        plt.ylim(0, slo_ttft[1] * 2.5)  # loosed SLO 기준
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    scatter_file = os.path.join(output_dir, f"performance_scatter_middle_{int(middle_ratio*100)}pct.png")
    plt.savefig(scatter_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Performance scatter plot saved to {scatter_file}")
    
    return stats, success_count



def create_token_distribution_plot(results, output_dir):
    """토큰 길이 분포 그래프 생성"""
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        print("No successful results for token distribution plot")
        return
    
    context_tokens = [r['context_tokens'] for r in successful_results]
    generated_tokens = [r['generated_tokens'] for r in successful_results]
    actual_generated_tokens = [r['actual_generated_tokens'] for r in successful_results]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Context tokens 분포
    ax1.hist(context_tokens, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('Context Tokens')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Context Tokens Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Generated tokens 분포
    ax2.hist(generated_tokens, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax2.set_xlabel('Generated Tokens')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Generated Tokens Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Actual generated tokens 분포
    ax3.hist(actual_generated_tokens, bins=50, alpha=0.7, color='red', edgecolor='black')
    ax3.set_xlabel('Actual Generated Tokens')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Actual Generated Tokens Distribution')
    ax3.grid(True, alpha=0.3)
    
    # Context vs Generated tokens scatter
    ax4.scatter(context_tokens, generated_tokens, alpha=0.5, s=30, color='purple')
    ax4.set_xlabel('Context Tokens')
    ax4.set_ylabel('Generated Tokens')
    ax4.set_title('Context vs Generated Tokens')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'token_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Token distribution plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Regenerate plots from benchmark JSON file")
    parser.add_argument("--json-file", 
                       default="benchmark_request.json",
                       help="Path to benchmark requests JSON file")
    parser.add_argument("--output-dir", 
                       default="benchmark_request",
                       help="Output directory for saving plots (default: benchmark_request)")
    parser.add_argument("--middle-ratio", 
                       type=float, 
                       default=0.7,
                       help="Ratio of middle data to use for performance stats (default: 0.7)")
    parser.add_argument("--plots", 
                       nargs='+',
                       default=['all'],
                       choices=['all', 'ttft', 'tpot', 'scatter', 'tokens'],
                       help="Which plots to generate (default: all)")

    parser.add_argument("--slo-tpot", 
                       type=float, 
                       nargs=2,
                       default=[0.116, 0.174],
                       help="SLO for TPOT [tight, loosed] (default: [0.116, 0.174])")
    parser.add_argument("--slo-ttft", 
                       type=float, 
                       nargs=2,
                       default=[1.30, 1.95],
                       help="SLO for TTFT [tight, loosed] (default: [1.30, 1.95])")

    args = parser.parse_args()
    
    # JSON 파일 로드
    try:
        results = load_benchmark_request(args.json_file)
    except FileNotFoundError:
        print(f"Error: JSON file '{args.json_file}' not found")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{args.json_file}'")
        return
    
    # 출력 디렉토리 생성
    print(f"Output directory: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 전역 데이터 복원
    ttft_graph, iter_tpot_graph, iter_num_prefill_graph, iter_num_decode_graph = reconstruct_global_data(results)
    
    # 플롯 생성
    plots_to_generate = args.plots
    if 'all' in plots_to_generate:
        plots_to_generate = ['ttft', 'tpot', 'scatter', 'tokens']

    print(f"\n--- Generating plots: {plots_to_generate} ---")
    
    if 'ttft' in plots_to_generate:
        save_ttft_plot(ttft_graph, 'iteration_step', 'ttft', 'ttft_graph', args.output_dir)
    
    if 'tpot' in plots_to_generate:
        save_tpot_plots(iter_tpot_graph, 'tpot_by_iteration', args.output_dir)

    
    if 'scatter' in plots_to_generate:
        create_performance_scatter_plots(results, args.output_dir, 
                                       middle_ratio=args.middle_ratio,
                                       slo_tpot=args.slo_tpot,
                                       slo_ttft=args.slo_ttft)
    
    if 'tokens' in plots_to_generate:
        create_token_distribution_plot(results, args.output_dir)
    
    print("\n--- Plot generation completed ---")

if __name__ == "__main__":
    main()