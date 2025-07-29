import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any

def calculate_latency_budgets(benchmark_file: str, output_file: str = "latency_budget.json", middle_ratio: float = 0.8):
    """
    Calculate latency budgets for requests within middle_ratio range and save to file.
    
    Args:
        benchmark_file: Path to benchmark_request.json file
        output_file: Path to output file for latency budgets
        middle_ratio: Ratio of middle data to use for SLO calculation and analysis
    """
    
    # Load benchmark data
    with open(benchmark_file, 'r') as f:
        benchmark_data = json.load(f)
    
    # Sort by request_id for consistent ordering
    benchmark_data.sort(key=lambda x: x['request_id'])
    
    # Apply middle_ratio filtering for SLO calculation
    if middle_ratio >= 1.0 or middle_ratio <= 0:
        print(f"Warning: Invalid middle_ratio {middle_ratio}, using all requests")
        filtered_data = benchmark_data
        start_idx = 0
        end_idx = len(benchmark_data)
    else:
        total_requests = len(benchmark_data)
        skip_count = int(total_requests * (1 - middle_ratio) / 2)
        start_idx = skip_count
        end_idx = total_requests - skip_count
        filtered_data = benchmark_data[start_idx:end_idx]
    
    print(f"Using requests {start_idx} to {end_idx-1} (total: {len(filtered_data)} requests) for SLO calculation")
    
    # Step 1: Calculate p90 tpot as SLO (using filtered successful requests)
    tpot_values = []
    for request in filtered_data:
        if request['success'] and 'tpot' in request and request['tpot'] > 0:
            tpot_values.append(request['tpot'])
    
    if not tpot_values:
        print("No valid tpot values found in filtered data!")
        return
    
    slo_tpot = float(np.percentile(tpot_values, 90))
    print(f"P90 TPOT (SLO) from filtered data: {slo_tpot:.6f} seconds")
    
    # Step 2: Calculate latency budgets for filtered requests only
    results = {
        'slo_tpot': slo_tpot,
        'middle_ratio': middle_ratio,
        'filter_range': f"{start_idx}-{end_idx-1}",
        'total_filtered_requests': len(filtered_data),
        'requests': []
    }
    
    skipped_count = 0
    token_mismatch_count = 0
    
    for request in filtered_data:
        if not request['success']:
            skipped_count += 1
            continue
            
        if 'itl' not in request or 'iteration_data' not in request:
            skipped_count += 1
            continue
            
        request_id = request['request_id']
        itl_list = request['itl']
        iteration_data = request['iteration_data']
        
        # 실제 생성된 토큰 수 사용
        if 'actual_generated_tokens' in request:
            generated_tokens = request['actual_generated_tokens']
        else:
            generated_tokens = request['generated_tokens']
        
        # 요청된 토큰 수와 실제 생성된 토큰 수 비교
        if 'requested_generated_tokens' in request:
            requested = request['requested_generated_tokens']
            if requested != generated_tokens:
                token_mismatch_count += 1
        
        # Check ITL data validity
        expected_itl_length = generated_tokens - 1
        actual_itl_length = len(itl_list) if itl_list else 0
        
        if not itl_list or actual_itl_length != expected_itl_length:
            print(f"Skipping request {request_id}: ITL length mismatch (expected {expected_itl_length}, got {actual_itl_length})")
            skipped_count += 1
            continue
        
        # iteration_data에서 global iteration 정보 추출
        if len(iteration_data) != generated_tokens:
            print(f"Skipping request {request_id}: iteration_data length mismatch (expected {generated_tokens}, got {len(iteration_data)})")
            skipped_count += 1
            continue
        
        # Global iteration 정보 추출 (첫 번째는 TTFT, 나머지는 ITL에 대응)
        global_iterations = [data['iteration_total'] for data in iteration_data]
        
        # Calculate latency budgets
        latency_budgets = []
        last_violation_step = -1
        
        for i, itl in enumerate(itl_list):
            if last_violation_step == -1:
                cumulative_itl = sum(itl_list[:i+1])
                steps_since_violation = i + 1
            else:
                cumulative_itl = sum(itl_list[last_violation_step+1:i+1])
                steps_since_violation = i - last_violation_step
            
            latency_budget = (slo_tpot * steps_since_violation) - cumulative_itl
            has_violation = bool(latency_budget < 0)
            
            if has_violation:
                last_violation_step = i
            
            # Global iteration: i+1번째 토큰이 생성된 시점의 iteration
            current_global_iteration = global_iterations[i+1] if i+1 < len(global_iterations) else None
            
            latency_budgets.append({
                'step': int(i),
                'token_index': int(i + 1),
                'global_iteration': int(current_global_iteration) if current_global_iteration is not None else None,
                'itl': float(itl),
                'latency_budget': float(latency_budget),
                'slo_violation': has_violation,
                'last_violation_step': int(last_violation_step),
                'steps_since_violation': int(steps_since_violation),
                'cumulative_itl': float(cumulative_itl)
            })
        
        total_violations = sum(1 for budget in latency_budgets if budget['slo_violation'])
        
        # Request의 global iteration 범위
        min_global_iter = min(global_iterations) if global_iterations else None
        max_global_iter = max(global_iterations) if global_iterations else None
        
        request_result = {
            'request_id': int(request_id),
            'requested_tokens': int(request.get('requested_generated_tokens', generated_tokens)),
            'actual_tokens': int(generated_tokens),
            'tpot': float(request['tpot']),
            'total_slo_violations': int(total_violations),
            'violation_rate': float(total_violations / len(latency_budgets)) if latency_budgets else 0.0,
            'min_global_iteration': int(min_global_iter) if min_global_iter is not None else None,
            'max_global_iteration': int(max_global_iter) if max_global_iter is not None else None,
            'latency_budgets': latency_budgets
        }
        
        results['requests'].append(request_result)
    
    # 전체 global iteration 범위 계산
    all_global_iterations = []
    for request in results['requests']:
        if request['min_global_iteration'] is not None:
            all_global_iterations.append(request['min_global_iteration'])
        if request['max_global_iteration'] is not None:
            all_global_iterations.append(request['max_global_iteration'])
    
    if all_global_iterations:
        results['global_iteration_range'] = {
            'min': int(min(all_global_iterations)),
            'max': int(max(all_global_iterations))
        }
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nLatency budgets calculated for filtered requests and saved to {output_file}")
    print(f"Total filtered requests: {len(filtered_data)}")
    print(f"Skipped requests: {skipped_count}")
    print(f"Analyzed requests: {len(results['requests'])}")
    if all_global_iterations:
        print(f"Global iteration range: {min(all_global_iterations)} - {max(all_global_iterations)}")
    
    return results


def analyze_latency_budgets(latency_budget_file: str = "benchmark_latency_budget.json", create_plots: bool = True):
    """
    Analyze latency budgets and create visualizations.
    """
    
    # Load latency budget data
    with open(latency_budget_file, 'r') as f:
        budget_data = json.load(f)
    
    all_requests = budget_data['requests']
    slo_tpot = budget_data['slo_tpot']
    middle_ratio = budget_data.get('middle_ratio', 'N/A')
    filter_range = budget_data.get('filter_range', 'N/A')
    global_iter_range = budget_data.get('global_iteration_range', {})
    
    print(f"\n=== Latency budget Analysis ===")
    print(f"Middle ratio used: {middle_ratio}")
    print(f"Filter range: {filter_range}")
    print(f"Total requests analyzed: {len(all_requests)}")
    print(f"SLO TPOT: {slo_tpot:.6f} seconds")
    if global_iter_range:
        print(f"Global iteration range: {global_iter_range.get('min', 'N/A')} - {global_iter_range.get('max', 'N/A')}")
    
    if not all_requests:
        print("No requests to analyze")
        return
    
    # Calculate statistics
    total_violations_across_requests = sum(r['total_slo_violations'] for r in all_requests)
    requests_with_violations = sum(1 for r in all_requests if r['total_slo_violations'] > 0)
    avg_violation_rate = np.mean([r['violation_rate'] for r in all_requests])
    
    print(f"\nViolation Statistics:")
    print(f"Requests with at least one SLO violation: {requests_with_violations} ({requests_with_violations/len(all_requests)*100:.1f}%)")
    print(f"Average violation rate per request: {avg_violation_rate*100:.1f}%")
    print(f"Total SLO violations: {total_violations_across_requests}")
    
    # budget statistics
    all_budgets = []
    for r in all_requests:
        all_budgets.extend([m['latency_budget'] for m in r['latency_budgets']])
    
    if all_budgets:
        print(f"\nLatency budget Statistics:")
        print(f"  Min budget: {min(all_budgets):.6f}")
        print(f"  Max budget: {max(all_budgets):.6f}")
        print(f"  Avg budget: {np.mean(all_budgets):.6f}")
        print(f"  Std budget: {np.std(all_budgets):.6f}")
        
        negative_budgets = [m for m in all_budgets if m < 0]
        positive_budgets = [m for m in all_budgets if m >= 0]
        
        print(f"\nbudget Distribution:")
        print(f"  Negative budgets (violations): {len(negative_budgets)} ({len(negative_budgets)/len(all_budgets)*100:.1f}%)")
        print(f"  Positive budgets: {len(positive_budgets)} ({len(positive_budgets)/len(all_budgets)*100:.1f}%)")
        
        if negative_budgets:
            print(f"  Avg negative budget: {np.mean(negative_budgets):.6f}")
        if positive_budgets:
            print(f"  Avg positive budget: {np.mean(positive_budgets):.6f}")
    
    violations_per_request = [r['total_slo_violations'] for r in all_requests]
    print(f"\nPer-Request Violation Distribution:")
    print(f"  Min violations: {min(violations_per_request)}")
    print(f"  Max violations: {max(violations_per_request)}")
    print(f"  Avg violations: {np.mean(violations_per_request):.2f}")
    print(f"  Median violations: {np.median(violations_per_request):.0f}")
    
    # Create visualizations
    if create_plots:
        import os
        from collections import Counter
        
        # Create output directory
        output_dir = "./benchmark_budget/"
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nCreating individual visualizations in {output_dir}...")
        
        plt.style.use('default')
        sns.set_palette("husl")
        
        # === 1. Timeline Analysis ===
        print("Creating timeline analysis...")
        
        global_iter_data = {}
        for request in all_requests:
            for budget in request['latency_budgets']:
                global_iter = budget.get('global_iteration')
                if global_iter is not None:
                    if global_iter not in global_iter_data:
                        global_iter_data[global_iter] = {'active': 0, 'violations': 0}
                    global_iter_data[global_iter]['active'] += 1
                    if budget['slo_violation']:
                        global_iter_data[global_iter]['violations'] += 1

        
        if global_iter_data:
            active_counts = [data['active'] for data in global_iter_data.values()]
            violation_counts = [data['violations'] for data in global_iter_data.values()]
            non_violation_counts = [data['active'] - data['violations'] for data in global_iter_data.values()]
            
            avg_decoding_reqs = np.mean(active_counts)
            avg_violating_reqs = np.mean(violation_counts)  
            avg_non_violating_reqs = np.mean(non_violation_counts)
            
            # Calculate violation rate statistics for steps that have violations
            steps_with_violations = []
            violation_rates = []
            
            for global_iter, data in global_iter_data.items():
                if data['violations'] > 0:  # Only consider steps with violations
                    violation_rate = data['violations'] / data['active']
                    steps_with_violations.append({
                        'global_iter': global_iter,
                        'active': data['active'],
                        'violations': data['violations'],
                        'violation_rate': violation_rate
                    })
                    violation_rates.append(violation_rate)
            
            print(f"\n=== Timeline Statistics ===")
            print(f"Total iteration steps analyzed: {len(global_iter_data)}")
            print(f"Average Decoding Reqs / iteration step: {avg_decoding_reqs:.2f}")
            print(f"Average Violating Reqs / iteration step: {avg_violating_reqs:.2f}")
            print(f"Average Non-violating Reqs / iteration step: {avg_non_violating_reqs:.2f}")
            print(f"Verification: {avg_violating_reqs:.2f} + {avg_non_violating_reqs:.2f} = {avg_violating_reqs + avg_non_violating_reqs:.2f} (should equal {avg_decoding_reqs:.2f})")
            
            # New statistics for steps with violations
            print(f"\n=== Violation Step Analysis ===")
            steps_with_violations_count = len(steps_with_violations)
            steps_without_violations_count = len(global_iter_data) - steps_with_violations_count
            
            print(f"Steps with violations: {steps_with_violations_count} ({steps_with_violations_count/len(global_iter_data)*100:.1f}%)")
            print(f"Steps without violations: {steps_without_violations_count} ({steps_without_violations_count/len(global_iter_data)*100:.1f}%)")
            
            if violation_rates:
                avg_violation_rate = np.mean(violation_rates) * 100
                min_violation_rate = min(violation_rates) * 100
                max_violation_rate = max(violation_rates) * 100
                median_violation_rate = np.median(violation_rates) * 100
                
                print(f"\nViolation Rate Statistics (among steps with violations):")
                print(f"  Average violation rate: {avg_violation_rate:.2f}%")
                print(f"  Minimum violation rate: {min_violation_rate:.2f}%")
                print(f"  Maximum violation rate: {max_violation_rate:.2f}%")
                print(f"  Median violation rate: {median_violation_rate:.2f}%")
                
                # Additional insights
                high_violation_steps = [step for step in steps_with_violations if step['violation_rate'] > 0.5]
                very_high_violation_steps = [step for step in steps_with_violations if step['violation_rate'] > 0.8]
                
                print(f"\nViolation Severity Analysis:")
                print(f"  Steps with >50% violation rate: {len(high_violation_steps)} ({len(high_violation_steps)/steps_with_violations_count*100:.1f}% of violation steps)")
                print(f"  Steps with >80% violation rate: {len(very_high_violation_steps)} ({len(very_high_violation_steps)/steps_with_violations_count*100:.1f}% of violation steps)")
                
                # Show some examples of high violation steps
                if high_violation_steps:
                    print(f"\nExample high violation steps:")
                    # Sort by violation rate and show top 3
                    high_violation_steps.sort(key=lambda x: x['violation_rate'], reverse=True)
                    for i, step in enumerate(high_violation_steps[:3]):
                        print(f"    Step {step['global_iter']}: {step['violations']}/{step['active']} requests violated ({step['violation_rate']*100:.1f}%)")
            else:
                print(f"\nNo steps with violations found!")
            
            # 추가 통계
            print(f"\nDetailed Statistics:")
            print(f"  Max concurrent decoding requests: {max(active_counts)}")
            print(f"  Min concurrent decoding requests: {min(active_counts)}")
            print(f"  Max violating requests per step: {max(violation_counts)}")
            print(f"  Steps with no violations: {sum(1 for v in violation_counts if v == 0)} ({sum(1 for v in violation_counts if v == 0)/len(violation_counts)*100:.1f}%)")
                    
                
        global_iters_with_data = sorted(global_iter_data.keys())
        
        if len(global_iters_with_data) > 1000:
            step = len(global_iters_with_data) // 500
            global_iters_with_data = global_iters_with_data[::step]
        
        active_counts = [global_iter_data[iter_num]['active'] for iter_num in global_iters_with_data]
        violation_counts = [global_iter_data[iter_num]['violations'] for iter_num in global_iters_with_data]
        
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(global_iters_with_data, active_counts, 'k-', linewidth=2, label='Active Requests', alpha=0.8)
        ax1.plot(global_iters_with_data, violation_counts, 'r-', linewidth=2, label='Violation Requests', alpha=0.8)
        ax1.fill_between(global_iters_with_data, violation_counts, alpha=0.3, color='red')
        ax1.set_xlabel('Global Iteration Step')
        ax1.set_ylabel('Number of Requests')
        ax1.set_title('Active vs Violation Requests per Global Iteration')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        if len(global_iters_with_data) > 10:
            step = max(1, len(global_iters_with_data) // 10)
            ax1.set_xticks(global_iters_with_data[::step])
            ax1.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '1_timeline_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # === 2. Violated Request Distribution ===
        print("Creating violation statistics with distribution...")
        
        # Count violations per request
        violation_counts = Counter(violations_per_request)
        
        # Calculate TPOT SLO violations
        tpot_slo_violations = 0
        tpot_slo_compliant = 0
        
        for request in all_requests:
            if request['tpot'] > slo_tpot:
                tpot_slo_violations += 1
            else:
                tpot_slo_compliant += 1
        
        fig2, (ax2_left, ax2_right) = plt.subplots(1, 2, figsize=(16, 8))
        
        # LEFT: Simple pie chart - latency budget violation vs no violation
        labels_left = ['Stalled Request', 'Non-Stalled Request']
        sizes_left = [requests_with_violations, len(all_requests) - requests_with_violations]
        colors_left = ['#ff6b6b', '#51cf66']  # Red for violations, Green for no violations
        explode_left = (0.05, 0)  # explode the violation slice
        
        wedges_left, texts_left, autotexts_left = ax2_left.pie(sizes_left, explode=explode_left, labels=labels_left, colors=colors_left,
                                               autopct='%1.1f%%', shadow=True, startangle=90, 
                                               textprops={'fontsize': 12, 'fontweight': 'bold'})
        ax2_left.set_title('Stalled Request vs Non-Stalled Request', fontsize=14, fontweight='bold')
        
        # Add count numbers to the pie chart labels and make autopct text larger
        for autotext, size in zip(autotexts_left, sizes_left):
            autotext.set_text(f'{autotext.get_text()}\n({size:,} requests)')
            autotext.set_fontsize(14)  # Larger font for percentage and count
            autotext.set_fontweight('bold')
        
        # RIGHT: TPOT SLO violation pie chart
        labels_right = ['Violated Request', 'Non-Violated Request']
        sizes_right = [tpot_slo_violations, tpot_slo_compliant]
        colors_right = ['#ff6b6b', '#51cf66']  # Same colors as left chart for consistency
        explode_right = (0.05, 0)  # explode the violation slice
        
        wedges_right, texts_right, autotexts_right = ax2_right.pie(sizes_right, explode=explode_right, labels=labels_right, colors=colors_right,
                                               autopct='%1.1f%%', shadow=True, startangle=90,
                                               textprops={'fontsize': 12, 'fontweight': 'bold'})
        ax2_right.set_title('Violated Request vs Non-Violated Request', fontsize=14, fontweight='bold')
        
        # Add count numbers to the pie chart labels and make autopct text larger
        for autotext, size in zip(autotexts_right, sizes_right):
            autotext.set_text(f'{autotext.get_text()}\n({size:,} requests)')
            autotext.set_fontsize(14)  # Larger font for percentage and count
            autotext.set_fontweight('bold')
        
        # Add summary statistics as text box for right chart
        tpot_violation_rate = (tpot_slo_violations / len(all_requests)) * 100
        stats_text = f"""TPOT SLO Statistics:
        SLO Threshold: {slo_tpot:.4f}s
        Total Requests: {len(all_requests):,}
        TPOT Violations: {tpot_slo_violations:,} ({tpot_violation_rate:.1f}%)
        TPOT Compliant: {tpot_slo_compliant:,} ({100-tpot_violation_rate:.1f}%)"""
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '2_violation_statistics_with_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print TPOT SLO statistics
        print(f"\n=== TPOT SLO Analysis ===")
        print(f"SLO TPOT Threshold: {slo_tpot:.6f} seconds")
        print(f"Requests with TPOT > SLO: {tpot_slo_violations} ({tpot_violation_rate:.1f}%)")
        print(f"Requests with TPOT ≤ SLO: {tpot_slo_compliant} ({100-tpot_violation_rate:.1f}%)")
        
        # Calculate some additional TPOT statistics
        tpot_values_all = [request['tpot'] for request in all_requests]
        violating_tpot_values = [request['tpot'] for request in all_requests if request['tpot'] > slo_tpot]
        compliant_tpot_values = [request['tpot'] for request in all_requests if request['tpot'] <= slo_tpot]
        
        print(f"\nTPOT Distribution:")
        print(f"  Overall TPOT - Min: {min(tpot_values_all):.6f}s, Max: {max(tpot_values_all):.6f}s, Avg: {np.mean(tpot_values_all):.6f}s")
        
        if violating_tpot_values:
            print(f"  Violating TPOT - Min: {min(violating_tpot_values):.6f}s, Max: {max(violating_tpot_values):.6f}s, Avg: {np.mean(violating_tpot_values):.6f}s")
        
        if compliant_tpot_values:
            print(f"  Compliant TPOT - Min: {min(compliant_tpot_values):.6f}s, Max: {max(compliant_tpot_values):.6f}s, Avg: {np.mean(compliant_tpot_values):.6f}s")
        
        # === 3 & 4. Enhanced Sample Request Analysis ===
        print("Finding middle section for enhanced sampling...")
        
        # Find middle section of global iterations (100 steps)
        if global_iter_range:
            min_global = global_iter_range['min']
            max_global = global_iter_range['max']
            middle_start = min_global + (max_global - min_global) // 2 - 50
            middle_end = middle_start + 100
            
            print(f"Analyzing middle section: {middle_start} - {middle_end}")
            
            # Find requests active in this middle section
            middle_section_requests = []
            for request in all_requests:
                request_global_iters = []
                for budget in request['latency_budgets']:
                    global_iter = budget.get('global_iteration')
                    if global_iter is not None and middle_start <= global_iter <= middle_end:
                        request_global_iters.append(global_iter)
                
                if request_global_iters:  # If request has any activity in middle section
                    # Add the range info to the request
                    request_copy = request.copy()
                    request_copy['middle_section_global_iters'] = sorted(request_global_iters)
                    middle_section_requests.append(request_copy)
            
            print(f"Found {len(middle_section_requests)} requests active in middle section")
            
            # Sample about 8 requests from middle section
            import random
            random.seed(42)
            
            sample_size = min(8, len(middle_section_requests))
            if sample_size > 0:
                sampled_requests = random.sample(middle_section_requests, sample_size)
                
                # Separate by violation status for plotting
                sampled_with_viols = [r for r in sampled_requests if r['total_slo_violations'] > 0]
                sampled_without_viols = [r for r in sampled_requests if r['total_slo_violations'] == 0]
                
                print(f"Sampled {len(sampled_without_viols)} requests without violations and {len(sampled_with_viols)} requests with violations")
                
                # === 3. Enhanced Requests WITHOUT Violations ===
                if sampled_without_viols:
                    print("Creating enhanced requests without violations plot...")
                    
                    fig3, ax3 = plt.subplots(figsize=(14, 8))
                    colors_no_viol = plt.cm.tab10(np.arange(len(sampled_without_viols)))
                    
                    for i, request in enumerate(sampled_without_viols):
                        # Extract global iterations and corresponding budgets for middle section
                        global_iters = []
                        budgets = []
                        
                        for budget in request['latency_budgets']:
                            global_iter = budget.get('global_iteration')
                            if global_iter is not None and middle_start <= global_iter <= middle_end:
                                global_iters.append(global_iter)
                                budgets.append(budget['latency_budget'])
                        
                        if global_iters:  # Only plot if there's data in the range
                            ax3.plot(global_iters, budgets, '-o', color=colors_no_viol[i], 
                                    alpha=0.8, linewidth=2, markersize=5,
                                    label=f"Req {request['request_id']} ({len(global_iters)} steps)")
                    
                    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=2, 
                               label='SLO Violation Threshold')
                    ax3.set_xlabel('Global Iteration Step')
                    ax3.set_ylabel('Latency budget (seconds)')
                    ax3.set_title(f'Requests WITHOUT Violations from Middle Section\n(Global Iter {middle_start}-{middle_end})', 
                                 fontweight='bold', color='green', fontsize=14)
                    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax3.grid(True, alpha=0.3)
                    ax3.set_xlim(middle_start, middle_end)
                    
                    # Highlight positive region
                    y_min, y_max = ax3.get_ylim()
                    if y_max > 0:
                        ax3.fill_between([middle_start, middle_end], 0, y_max, alpha=0.1, color='green')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, '3_requests_without_violations.png'), 
                               dpi=300, bbox_inches='tight')
                    plt.close()
                
                # === 4. Enhanced Requests WITH Violations ===
                if sampled_with_viols:
                    print("Creating enhanced requests with violations plot...")
                    
                    fig4, ax4 = plt.subplots(figsize=(14, 8))
                    colors_with_viol = plt.cm.tab10(np.arange(len(sampled_with_viols)))
                    
                    for i, request in enumerate(sampled_with_viols):
                        # Extract global iterations and corresponding budgets for middle section
                        global_iters = []
                        budgets = []
                        violation_global_iters = []
                        violation_budgets = []
                        
                        for budget in request['latency_budgets']:
                            global_iter = budget.get('global_iteration')
                            if global_iter is not None and middle_start <= global_iter <= middle_end:
                                global_iters.append(global_iter)
                                budgets.append(budget['latency_budget'])
                                
                                if budget['slo_violation']:
                                    violation_global_iters.append(global_iter)
                                    violation_budgets.append(budget['latency_budget'])
                        
                        if global_iters:  # Only plot if there's data in the range
                            ax4.plot(global_iters, budgets, '-o', color=colors_with_viol[i], 
                                    alpha=0.8, linewidth=2, markersize=5,
                                    label=f"Req {request['request_id']} ({len([m for m in request['latency_budgets'] if m.get('global_iteration') and middle_start <= m.get('global_iteration') <= middle_end and m['slo_violation']])} violations in range)")
                            
                            # Highlight violation points
                            if violation_global_iters:
                                ax4.scatter(violation_global_iters, violation_budgets, color='red', s=120, 
                                           alpha=0.9, edgecolors='darkred', linewidth=2, zorder=5)
                    
                    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=2, 
                               label='SLO Violation Threshold')
                    ax4.set_xlabel('Global Iteration Step')
                    ax4.set_ylabel('Latency budget (seconds)')
                    ax4.set_title(f'Requests WITH Violations from Middle Section\n(Global Iter {middle_start}-{middle_end})', 
                                 fontweight='bold', color='red', fontsize=14)
                    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax4.grid(True, alpha=0.3)
                    ax4.set_xlim(middle_start, middle_end)
                    
                    # Highlight negative region
                    y_min, y_max = ax4.get_ylim()
                    if y_min < 0:
                        ax4.fill_between([middle_start, middle_end], y_min, 0, alpha=0.1, color='red')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, '4_requests_with_violations.png'), 
                               dpi=300, bbox_inches='tight')
                    plt.close()
                
                # If only one type exists, create a combined plot
                if len(sampled_without_viols) == 0 or len(sampled_with_viols) == 0:
                    print("Creating combined sample requests plot...")
                    
                    fig_combined, ax_combined = plt.subplots(figsize=(14, 8))
                    colors_combined = plt.cm.tab10(np.arange(len(sampled_requests)))
                    
                    for i, request in enumerate(sampled_requests):
                        # Extract global iterations and corresponding budgets for middle section
                        global_iters = []
                        budgets = []
                        violation_global_iters = []
                        violation_budgets = []
                        
                        for budget in request['latency_budgets']:
                            global_iter = budget.get('global_iteration')
                            if global_iter is not None and middle_start <= global_iter <= middle_end:
                                global_iters.append(global_iter)
                                budgets.append(budget['latency_budget'])
                                
                                if budget['slo_violation']:
                                    violation_global_iters.append(global_iter)
                                    violation_budgets.append(budget['latency_budget'])
                        
                        if global_iters:  # Only plot if there's data in the range
                            ax_combined.plot(global_iters, budgets, '-o', color=colors_combined[i], 
                                            alpha=0.8, linewidth=2, markersize=5,
                                            label=f"Req {request['request_id']} ({request['total_slo_violations']} viol)")
                            
                            if violation_global_iters:
                                ax_combined.scatter(violation_global_iters, violation_budgets, color='red', s=120, 
                                                   alpha=0.9, edgecolors='darkred', linewidth=2, zorder=5)
                    
                    ax_combined.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=2, 
                                       label='SLO Violation Threshold')
                    ax_combined.set_xlabel('Global Iteration Step')
                    ax_combined.set_ylabel('Latency budget (seconds)')
                    ax_combined.set_title(f'Sample Requests from Middle Section\n(Global Iter {middle_start}-{middle_end})', 
                                         fontweight='bold', fontsize=14)
                    ax_combined.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax_combined.grid(True, alpha=0.3)
                    ax_combined.set_xlim(middle_start, middle_end)
                    
                    # Add background shading
                    y_min, y_max = ax_combined.get_ylim()
                    if y_min < 0:
                        ax_combined.fill_between([middle_start, middle_end], y_min, 0, alpha=0.1, color='red')
                    if y_max > 0:
                        ax_combined.fill_between([middle_start, middle_end], 0, y_max, alpha=0.1, color='green')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, '3_4_combined_sample_requests.png'), 
                               dpi=300, bbox_inches='tight')
                    plt.close()
        
        
                # === 5. Stalled Time vs Violation Count Analysis ===
                print("Creating stalled time vs violation count analysis...")

                # Calculate stalled time for each request
                stalled_time_data = []
                for request in all_requests:
                    total_violations = request['total_slo_violations']
                    if total_violations > 0:  # Only consider requests with violations
                        total_stalled_time = 0
                        for budget in request['latency_budgets']:
                            if budget['slo_violation']:
                                # Stalled time is the absolute value of negative latency budget
                                total_stalled_time += abs(budget['latency_budget'])
                        
                        stalled_time_data.append({
                            'request_id': request['request_id'],
                            'violation_count': total_violations,
                            'total_stalled_time': total_stalled_time,
                            'avg_stall_per_violation': total_stalled_time / total_violations
                        })

                if stalled_time_data:
                    fig5, ax5 = plt.subplots(figsize=(12, 8))
                    
                    violation_counts = [data['violation_count'] for data in stalled_time_data]
                    stalled_times = [data['total_stalled_time'] * 1000 for data in stalled_time_data]  # Convert to ms
                    
                    # Create scatter plot
                    scatter = ax5.scatter(violation_counts, stalled_times, alpha=0.6, s=50, 
                                        c=violation_counts, cmap='Reds', edgecolors='black', linewidth=0.5)
                    
                    ax5.set_xlabel('Number of SLO Violations per Request')
                    ax5.set_ylabel('Total Stalled Time (ms)')
                    ax5.set_title('Stalled Time vs SLO Violation Count\n(Only Requests with Violations)', 
                                fontweight='bold', fontsize=14)
                    ax5.grid(True, alpha=0.3)
                    
                    # Add colorbar
                    cbar = plt.colorbar(scatter, ax=ax5)
                    cbar.set_label('Violation Count', rotation=270, labelpad=15)
                    
                    # Calculate and display statistics
                    total_requests_with_violations = len(stalled_time_data)
                    violation_percentage = (total_requests_with_violations / len(all_requests)) * 100
                    
                    total_stalled_time_ms = sum(stalled_times)
                    avg_stalled_time_per_request = total_stalled_time_ms / total_requests_with_violations
                    total_violations = sum(violation_counts)
                    avg_stall_per_violation = total_stalled_time_ms / total_violations
                    
                    stats_text = f"""Statistics:
                Requests with violations: {total_requests_with_violations:,} ({violation_percentage:.1f}%)
                Total stalled time: {total_stalled_time_ms:.1f} ms
                Avg stalled time per violating request: {avg_stalled_time_per_request:.2f} ms
                Avg stall per violation: {avg_stall_per_violation:.2f} ms
                Total violations: {total_violations:,}"""
                    
                    ax5.text(0.02, 0.98, stats_text, transform=ax5.transAxes, 
                            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.8),
                            verticalalignment='top', horizontalalignment='left', fontsize=10,
                            fontfamily='monospace')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, '5_stalled_time_vs_violations.png'), 
                            dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    # Print detailed statistics
                    print(f"\n=== Stalled Time Analysis ===")
                    print(f"Requests with violations: {total_requests_with_violations:,} out of {len(all_requests):,} ({violation_percentage:.1f}%)")
                    print(f"Total stalled time across all violating requests: {total_stalled_time_ms:.1f} ms")
                    print(f"Average stalled time per violating request: {avg_stalled_time_per_request:.2f} ms")
                    print(f"Average stall time per individual violation: {avg_stall_per_violation:.2f} ms")
                    print(f"Total number of violations: {total_violations:,}")
                    
                    # Additional statistics
                    stalled_times_only = [data['total_stalled_time'] * 1000 for data in stalled_time_data]
                    stall_per_violation_list = [data['avg_stall_per_violation'] * 1000 for data in stalled_time_data]
                    
                    print(f"\nStalled Time Distribution:")
                    print(f"  Min stalled time: {min(stalled_times_only):.2f} ms")
                    print(f"  Max stalled time: {max(stalled_times_only):.2f} ms")
                    print(f"  Median stalled time: {np.median(stalled_times_only):.2f} ms")
                    print(f"  Std stalled time: {np.std(stalled_times_only):.2f} ms")
                    
                    print(f"\nStall Per Violation Distribution:")
                    print(f"  Min stall per violation: {min(stall_per_violation_list):.2f} ms")
                    print(f"  Max stall per violation: {max(stall_per_violation_list):.2f} ms")
                    print(f"  Median stall per violation: {np.median(stall_per_violation_list):.2f} ms")
                    print(f"  Std stall per violation: {np.std(stall_per_violation_list):.2f} ms")
                else:
                    print("No requests with violations found for stalled time analysis")
        
        
        print(f"\nAll plots saved to {output_dir}:")
        print("  - 1_timeline_analysis.png")
        print("  - 2_violation_count_distribution.png") 
        print("  - 3_requests_without_violations.png")
        print("  - 4_requests_with_violations.png")
        print("  - 5_stalled_time_vs_violations.png")
    
    return {
        'analyzed_requests': len(all_requests),
        'requests_with_violations': requests_with_violations,
        'violation_percentage': requests_with_violations/len(all_requests)*100,
        'avg_violation_rate': avg_violation_rate*100,
        'total_violations': total_violations_across_requests,
        'global_iteration_range': global_iter_range,
        'violation_distribution': dict(Counter(violations_per_request))
    }



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calculate and analyze latency budgets")
    parser.add_argument("--mode", choices=['calculate', 'analyze', 'both'], default='both',
                       help="Mode: calculate budgets, analyze, or both")
    parser.add_argument("--benchmark-file", default="benchmark_request.json",
                       help="Path to benchmark results file (for calculate mode)")
    parser.add_argument("--budget-file", default="benchmark_budget.json",
                       help="Path to latency budget file")
    parser.add_argument("--middle-ratio", type=float, default=0.7,
                       help="Ratio of middle data to use for SLO calculation (default: 0.8)")
    parser.add_argument("--no-plots", action="store_true",
                       help="Skip creating visualization plots")
    
    args = parser.parse_args()
    
    if args.mode in ['calculate', 'both']:
        calculate_latency_budgets(args.benchmark_file, args.budget_file, args.middle_ratio)
    
    if args.mode in ['analyze', 'both']:
        analyze_latency_budgets(args.budget_file, create_plots=not args.no_plots)