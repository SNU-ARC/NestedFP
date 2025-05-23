import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

# CSV 파일에서 데이터를 읽어와 그래프를 생성하는 함수
def create_graph_from_csv(csv_file):
    # 사용할 batch size
    x = [32, 64, 128, 256, 512]
    
    # 모델 순서
    model_order = ["llama3.1_8b", "mistral_nemo_12b", "mistral_small_24b","phi_4"]
    
    # 하단 모델 이름용 라벨
    model_labels = {
        "llama3.1_8b": "LLaMA3.1(8B)",
        "mistral_nemo_12b": "Mistral-Nemo(12B)",
        "phi_4": "Phi-4(14B)",
        "mistral_small_24b": "Mistral-Small(24B)"
    }
    
    colors = {
        "torch_fp16": "darkred",
        "torch_fp8": "orange",
        "dualfp_fp16": "darkblue",
        "dualfp_fp8": "lightblue"
    }
    
    # CSV 파일 읽기
    df = pd.read_csv(csv_file)
    
    # 모델별 데이터 추출
    models = {}
    for i, model_key in enumerate(model_order):
        models[model_key] = {
            "torch_fp16": df.iloc[:, i*4].tolist(),
            "torch_fp8": df.iloc[:, i*4 + 1].tolist(),
            "dualfp_fp16": df.iloc[:, i*4 + 2].tolist(),
            "dualfp_fp8": df.iloc[:, i*4 + 3].tolist()
        }
    
    # figure 생성
    fig, axes = plt.subplots(1, 4, figsize=(24, 4), constrained_layout=True)
    
    # 각 그래프 그리기
    for i, (model_key, label) in enumerate(model_labels.items()):
        model_data = models[model_key]

        y1 = model_data["torch_fp16"]
        y2 = model_data["torch_fp8"]
        y3 = model_data["dualfp_fp16"]
        y4 = model_data["dualfp_fp8"]

        local_max_y = max(max(y1), max(y2), max(y3), max(y4))

        lw = 1.5
        ax = axes[i]
        ax.plot(x, y1, marker='o', markersize=10, markerfacecolor='none', color=colors["torch_fp16"], label="Torch FP16", linewidth=lw)
        ax.plot(x, y2, marker='s', markersize=10, color=colors["torch_fp8"], label="Torch FP8", linewidth=lw)
        ax.plot(x, y3, marker='x', markersize=10, color=colors["dualfp_fp16"], label="NestedFP16", linewidth=lw)
        ax.plot(x, y4, marker='o', markersize=10, color=colors["dualfp_fp8"], label="NestedFP8", linewidth=lw)

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

    # 범례 마커도 동일하게 추가
    fig.legend(
        handles=[
            plt.Line2D([], [], color=colors["torch_fp16"], marker='o', markerfacecolor='none', label="Torch FP16", markersize=10, linewidth=lw),
            plt.Line2D([], [], color=colors["torch_fp8"], marker='s', label="Torch FP8", markersize=10, linewidth=lw),
            plt.Line2D([], [], color=colors["dualfp_fp16"], marker='x', label="NestedFP16", markersize=10, linewidth=lw),
            plt.Line2D([], [], color=colors["dualfp_fp8"], marker='o', label="NestedFP8", markersize=10, linewidth=lw),
        ],
        loc='upper center',
        bbox_to_anchor=(0.5, 1.35),
        ncol=4,
        fontsize=28,
        frameon=False
    )

    # 파일명에서 확장자 제거하고 저장
    filename_without_ext = os.path.splitext(os.path.basename(csv_file))[0]
    plt.savefig(f"e2e_performance_fp8_{filename_without_ext}.pdf", dpi=400, bbox_inches='tight')
    plt.show()

# 여러 CSV 파일을 처리하는 함수
def process_multiple_csvs(csv_files):
    """여러 CSV 파일을 처리하여 각각 별도의 그래프 생성"""
    for csv_file in csv_files:
        print(f"Processing {csv_file}...")
        create_graph_from_csv(csv_file)
        print(f"Graph saved for {csv_file}")

# 사용 예시:

# 방법 1: 특정 CSV 파일들 지정
csv_files = ["32_32.csv", "32_512.csv", "1024_32.csv", "1024_512.csv"]  # 실제 파일명으로 변경
process_multiple_csvs(csv_files)

# 방법 2: 현재 디렉토리의 모든 CSV 파일 처리
# csv_files = glob.glob("*.csv")
# csv_files.sort()  # 파일명 순서대로 정렬
# process_multiple_csvs(csv_files)

# 방법 3: 단일 CSV 파일 처리
# create_graph_from_csv("your_data.csv")

# 예시: 특정 패턴의 CSV 파일들만 처리
# csv_files = glob.glob("experiment_*.csv")
# process_multiple_csvs(csv_files)