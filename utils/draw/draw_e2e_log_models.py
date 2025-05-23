import matplotlib.pyplot as plt


# 사용할 batch size
x = [32, 64, 128, 256, 512]

# 모델별 throughput (toks/s)
models = {
    "llama3.1_8b": {
        "torch_fp16": [3104.373033, 5465.282943, 8944.713194, 13442.94832, 16565.32567],
        "torch_fp8": [3968.293952, 6822.998061, 11193.62788, 15989.31423, 20483.95097],
        "dualfp_fp16": [3025.65891, 5320.627572, 8934.726469, 12939.51354, 16711.7989],
        "dualfp_fp8":  [3879.163237, 6699.495395, 10907.57359, 16012.24081, 20246.04842],
    },
    "mistral_nemo_12b": {
        "torch_fp16": [1432.937751, 2684.5112, 4751.133028, 7283.539847, 9520.882347],
        "torch_fp8": [1432.937751, 2684.5112, 4751.133028, 7283.539847, 9520.882347],
        "dualfp_fp16": [1367.458825, 2536.65719, 4490.822681, 6953.049429, 9392.706297],
        "dualfp_fp8":  [1862.062048, 3398.495982, 5860.116101, 9053.650077, 12258.40251],
    },
    "phi_4": {
        "torch_fp16": [1924.031183, 3462.936889, 5710.328157, 8388.456321, 9187.108498],
        "torch_fp8": [1924.031183, 3462.936889, 5710.328157, 8388.456321, 9187.108498],
        "dualfp_fp16": [1853.561722, 3291.628134, 5472.062879, 8235.512449, 9350.333366],
        "dualfp_fp8":  [2513.017983, 4388.829959, 7071.452847, 10291.92274, 11576.27596],
    },
    "mistral_small_24b": {
        "torch_fp16": [861.6991766, 1640.77707, 2960.529163, 4747.861881, 5319.922],
        "torch_fp8": [861.6991766, 1640.77707, 2960.529163, 4747.861881, 5319.922],
        "dualfp_fp16": [839.1832294, 1574.350475, 2824.415844, 4463.405518, 5244.43137],
        "dualfp_fp8":  [1297.172415, 2411.954005, 4242.274904, 6753.534196, 8042.69577],
    },
}

# 하단 모델 이름용 라벨
model_labels = {
    "llama3.1_8b": "LLaMA3.1(8B)",
    "mistral_nemo_12b": "Mistral-Nemo(12B)",
    "phi_4": "Phi-4(14B)",
    "mistral_small_24b": "Mistral-Small(24B)"
}

colors = {
    "torch_fp16": "darkred",
    "dualfp_fp8": "#009e73",
    "dualfp_fp16": "darkblue"
}

# figure 생성
fig, axes = plt.subplots(1, 4, figsize=(24, 4), constrained_layout=True)

# 각 그래프 그리기
for i, (model_key, label) in enumerate(model_labels.items()):
    model_data = models[model_key]

    y1 = model_data["torch_fp16"]
    y2 = model_data["dualfp_fp16"]
    y3 = model_data["dualfp_fp8"]

    local_max_y = max(max(y1), max(y2), max(y3))

    lw = 1.5
    ax = axes[i]
    ax.plot(x, y1, marker='o', markersize=10, markerfacecolor='none', color=colors["torch_fp16"], label="FP16", linewidth = lw)
    ax.plot(x, y2, marker='x', markersize=10, color=colors["dualfp_fp16"], label="NestedFP16", linewidth = lw)  # X 마커
    ax.plot(x, y3, marker='o', markersize=10, color=colors["dualfp_fp8"], label="NestedFP8", linewidth=lw)  # 꽉 찬 원

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

# 범례 마커도 동일하게 수정
fig.legend(
    handles=[
        plt.Line2D([], [], color=colors["torch_fp16"], marker='o', markerfacecolor='none', label="FP16", markersize=10),  # 빈 원
        plt.Line2D([], [], color=colors["dualfp_fp16"], marker='x', label="NestedFP16", markersize=10),  # X 마커
        plt.Line2D([], [], color=colors["dualfp_fp8"], marker='o', label="NestedFP8", markersize=10),  # 꽉 찬 원
    ],
    loc='upper center',
    bbox_to_anchor=(0.5, 1.3),
    ncol=3,
    fontsize=32,
    frameon=False
)


plt.savefig("e2e_performance.pdf", dpi=400, bbox_inches='tight')

