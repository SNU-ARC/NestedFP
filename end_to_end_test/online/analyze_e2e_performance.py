import json
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
import glob

# ============================================================================
# Part 1: CSV 생성 (create_csv_from_results.py)
# ============================================================================

def load_json_results(filepath):
    """JSON 파일을 읽어서 결과를 반환"""
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_throughput_data(json_data, input_len, output_len):
    """
    특정 input/output 조합에 대한 batch size별 throughput 추출
    
    Args:
        json_data: JSON 파일에서 읽은 데이터
        input_len: 찾고자 하는 input length
        output_len: 찾고자 하는 output length
    
    Returns:
        dict: {batch_size: throughput} 형태의 딕셔너리
    """
    throughput_dict = {}
    
    for result in json_data['results']:
        if result['input_len'] == input_len and result['output_len'] == output_len:
            batch_size = result['batch_size']
            throughput = result['throughput_tokens_per_sec']
            throughput_dict[batch_size] = throughput
    
    return throughput_dict

def create_csv_for_config(results_dir, input_len, output_len, output_csv):
    """
    특정 input/output 조합에 대한 CSV 파일 생성
    
    Args:
        results_dir: 결과 파일이 있는 디렉토리
        input_len: input length
        output_len: output length
        output_csv: 출력할 CSV 파일명
    """
    
    # 모델 매핑 (베이스 파일명 -> 모델 키)
    model_mapping = {
        "throughput_sweep_Llama-3.1-8B": "llama3.1_8b",
        "throughput_sweep_Mistral-Nemo-Base-2407": "mistral_nemo_12b",
        "throughput_sweep_Mistral-Small-24B-Base-2501": "mistral_small_24b",
        "throughput_sweep_phi-4": "phi_4"
    }
    
    # 모델 순서 (CSV 열 순서)
    model_order = ["llama3.1_8b", "mistral_nemo_12b", "mistral_small_24b", "phi_4"]
    
    # batch size 순서
    batch_sizes = [32, 64, 128, 256, 512]
    
    # 데이터를 저장할 딕셔너리
    data = {batch_size: [] for batch_size in batch_sizes}
    
    # 각 모델에 대해 데이터 추출
    for model_key in model_order:
        # 베이스 파일명 찾기
        base_filename = None
        for filename, key in model_mapping.items():
            if key == model_key:
                base_filename = filename
                break
        
        if base_filename is None:
            print(f"Warning: No filename found for model {model_key}")
            continue
        
        # FP16 (_False.json) 및 NestedFP (_True.json) 파일 경로
        fp16_path = os.path.join(results_dir, f"{base_filename}_False.json")
        NestedFP_path = os.path.join(results_dir, f"{base_filename}_True.json")
        
        if not os.path.exists(fp16_path):
            print(f"Warning: File not found: {fp16_path}")
            continue
        if not os.path.exists(NestedFP_path):
            print(f"Warning: File not found: {NestedFP_path}")
            continue
        
        # JSON 파일 로드
        fp16_data = load_json_results(fp16_path)
        NestedFP_data = load_json_results(NestedFP_path)
        
        # Throughput 추출
        fp16_throughput = extract_throughput_data(fp16_data, input_len, output_len)
        NestedFP_throughput = extract_throughput_data(NestedFP_data, input_len, output_len)
        
        # 각 batch size에 대해 데이터 추가
        for batch_size in batch_sizes:
            fp16_val = fp16_throughput.get(batch_size, 0)
            NestedFP_val = NestedFP_throughput.get(batch_size, 0)
            data[batch_size].extend([fp16_val, NestedFP_val])
    
    # DataFrame 생성
    df = pd.DataFrame(data).T
    
    # 열 이름 설정
    column_names = []
    model_labels = {
        "llama3.1_8b": "Llama-3.1-8B",
        "mistral_nemo_12b": "Mistral-Nemo-12B",
        "mistral_small_24b": "Mistral-Small-24B",
        "phi_4": "Phi-4"
    }
    
    for model_key in model_order:
        model_label = model_labels[model_key]
        column_names.append(f"{model_label}_FP16")
        column_names.append(f"{model_label}_NestedFP")
    
    df.columns = column_names
    
    # CSV 저장
    df.to_csv(output_csv, index=False)
    print(f"Created: {output_csv}")
    print(f"  Input length: {input_len}, Output length: {output_len}")
    print(f"  Shape: {df.shape}")
    print()

def create_all_csvs(results_dir, output_dir="."):
    """
    모든 input/output 조합에 대한 CSV 파일 생성
    
    Args:
        results_dir: 결과 파일이 있는 디렉토리 (_False.json, _True.json 파일들)
        output_dir: CSV 파일을 저장할 디렉토리
    """
    
    # input/output 조합
    configs = [
        # (32, 32),
        # (32, 512),
        (128, 32),
        (256, 32),
        (512, 32),
        (1024, 32),
        (128, 512),
        (256, 512),
        (512, 512),
        (1024, 512)
    ]
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("STEP 1: Creating CSV files from JSON results")
    print("="*80)
    print(f"  Results directory: {results_dir}")
    print(f"  Output directory: {output_dir}")
    print()
    
    # 각 조합에 대해 CSV 생성
    for input_len, output_len in configs:
        output_csv = os.path.join(output_dir, f"{input_len}_{output_len}.csv")
        create_csv_for_config(results_dir, input_len, output_len, output_csv)
    
    print(f"✓ All CSV files created successfully in {output_dir}!")
    print()


# ============================================================================
# Part 2: 그래프 생성 (draw_e2e_models.py)
# ============================================================================

def create_graph_from_csv(csv_file):
    """CSV 파일에서 데이터를 읽어와 그래프를 생성하는 함수"""
    # 사용할 batch size
    x = [32, 64, 128, 256, 512]
    
    # 모델 순서
    model_order = ["llama3.1_8b", "mistral_nemo_12b", "mistral_small_24b", "phi_4"]
    
    # 하단 모델 이름용 라벨
    model_labels = {
        "llama3.1_8b": "LLaMA3.1(8B)",
        "mistral_nemo_12b": "Mistral-Nemo(12B)",
        "phi_4": "Phi-4(14B)",
        "mistral_small_24b": "Mistral-Small(24B)"
    }
    
    colors = {
        "torch_fp16": "darkred",
        "dualfp_fp16": "darkblue"
    }
    
    # CSV 파일 읽기
    df = pd.read_csv(csv_file)
    
    # 모델별 데이터 추출 (torch_fp16과 dualfp_fp16만)
    models = {}
    for i, model_key in enumerate(model_order):
        models[model_key] = {
            "torch_fp16": df.iloc[:, i*2].tolist(),
            "dualfp_fp16": df.iloc[:, i*2 + 1].tolist()
        }
    
    # figure 생성
    fig, axes = plt.subplots(1, 4, figsize=(24, 4), constrained_layout=True)
    
    # 각 그래프 그리기
    for i, (model_key, label) in enumerate(model_labels.items()):
        model_data = models[model_key]

        y1 = model_data["torch_fp16"]
        y2 = model_data["dualfp_fp16"]

        local_max_y = max(max(y1), max(y2))

        lw = 1.5
        ax = axes[i]
        ax.plot(x, y1, marker='o', markersize=10, markerfacecolor='none', color=colors["torch_fp16"], label="FP16", linewidth=lw)
        ax.plot(x, y2, marker='x', markersize=10, color=colors["dualfp_fp16"], label="NestedFP", linewidth=lw)

        ax.set_xlabel("Batch Size", fontsize=24)
        if i == 0:
            ax.set_ylabel("Throughput (toks/s)", fontsize=24)

        ax.set_xlim(28, 600)
        ax.set_ylim(0, local_max_y * 1.05)
        ax.set_xscale("log", base=2)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in x])
        ax.tick_params(axis='x', labelsize=22)
        ax.tick_params(axis='y', labelsize=22)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        ax.set_title("", fontsize=1)
        ax.text(0.5, -0.35, label, fontsize=32, ha='center', va='top', transform=ax.transAxes)

    # 범례
    fig.legend(
        handles=[
            plt.Line2D([], [], color=colors["torch_fp16"], marker='o', markerfacecolor='none', label="FP16", markersize=10, linewidth=lw),
            plt.Line2D([], [], color=colors["dualfp_fp16"], marker='x', label="NestedFP", markersize=10, linewidth=lw),
        ],
        loc='upper center',
        bbox_to_anchor=(0.5, 1.3),
        ncol=2,
        fontsize=32,
        frameon=False
    )

    # 파일명에서 확장자 제거하고 저장
    filename_without_ext = os.path.splitext(os.path.basename(csv_file))[0]
    plt.savefig(f"e2e_performance_{filename_without_ext}.pdf", dpi=400, bbox_inches='tight')
    plt.close()
    
    # 각 모델별 평균 오버헤드 계산 및 출력
    print(f"\n=== Overhead Analysis for {filename_without_ext} ===")
    for model_key, label in model_labels.items():
        model_data = models[model_key]
        y1 = model_data["torch_fp16"]
        y2 = model_data["dualfp_fp16"]
        
        # 각 batch size별 오버헤드 계산 (음수면 느려진 것, 양수면 빨라진 것)
        overheads = []
        for fp16_val, nested_val in zip(y1, y2):
            if fp16_val > 0:  # 0으로 나누는 것 방지
                overhead_percent = ((nested_val - fp16_val) / fp16_val) * 100
                overheads.append(overhead_percent)
        
        # 평균 오버헤드
        if overheads:
            avg_overhead = sum(overheads) / len(overheads)
            if avg_overhead >= 0:
                print(f"  {label}: NestedFP is {avg_overhead:.2f}% faster on average")
            else:
                print(f"  {label}: NestedFP has {abs(avg_overhead):.2f}% overhead on average")
            
            # 상세 정보 (각 batch size별)
            print(f"    Batch sizes: ", end="")
            for bs, oh in zip(x, overheads):
                if oh >= 0:
                    print(f"{bs}(+{oh:.1f}%) ", end="")
                else:
                    print(f"{bs}({oh:.1f}%) ", end="")
            print()
    print()

def process_multiple_csvs(csv_files):
    """여러 CSV 파일을 처리하여 각각 별도의 그래프 생성"""
    print("="*80)
    print("STEP 2: Creating graphs from CSV files")
    print("="*80)
    print()
    
    for csv_file in csv_files:
        print(f"\n{'='*60}")
        print(f"Processing {csv_file}...")
        print(f"{'='*60}")
        create_graph_from_csv(csv_file)
        print(f"✓ Graph saved: e2e_performance_{os.path.splitext(csv_file)[0]}.pdf")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    # 경로 설정
    results_dir = "/home2/omin/NestedFP/end_to_end_test/online/max_batched_tokens=8192"
    output_dir = "."  # 현재 디렉토리에 저장
    
    print("\n" + "="*80)
    print("E2E Performance Analysis Pipeline")
    print("="*80)
    print()
    
    # Step 1: CSV 파일 생성
    create_all_csvs(results_dir, output_dir)
    
    # 생성된 CSV 파일 목록
    print("\nGenerated CSV files:")
    csv_files_list = []
    for csv_file in sorted(Path(output_dir).glob("*_*.csv")):
        print(f"  - {csv_file.name}")
        csv_files_list.append(csv_file.name)
    print()
    
    # Step 2: 그래프 생성
    if csv_files_list:
        process_multiple_csvs(csv_files_list)
    else:
        print("Warning: No CSV files found!")
    
    print("\n" + "="*80)
    print("Pipeline completed successfully!")
    print("="*80)