import json
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import argparse
from matplotlib.ticker import LogLocator, FuncFormatter
import matplotlib.gridspec as gridspec

def load_json_robust(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류: {e}")
        print("수동 처리 시도 중...")
        with open(file_path, 'r') as f:
            content = f.read()
        content = re.sub(r':\s*NaN', ': null', content)
        content = re.sub(r',(\s*[\]}])', r'\1', content)
        content = content.replace('\'', '"')
        try:
            return json.loads(content)
        except json.JSONDecodeError as e2:
            print(f"수동 처리 후에도 파싱 실패: {e2}")
            raise e2

def plot_weight_distributions(model_names=None):
    if model_names is None:
        model_names = [
            'Llama-3.1-8B',
            'Mistral-Nemo',
            'Mistral-Small',
            'Phi-4'
        ]
    
    plt.style.use('default')
    TH = 1.75
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    model_stats = []
    all_data = []

    for i, model_name in enumerate(model_names):
        json_file = f"{model_name}_stats.json"
        try:
            model_data = load_json_robust(json_file)
            bin_centers = np.array(model_data['histogram']['bin_centers'])
            hist_values = np.array(model_data['histogram']['hist_values'])
            valid_indices = np.where(hist_values > 0)[0]
            bin_centers = bin_centers[valid_indices]
            hist_values = hist_values[valid_indices]
            all_data.append((model_name, bin_centers, hist_values, model_data))
            model_stats.append({
                'name': model_name,
                'min': model_data['weight_min'],
                'max': model_data['weight_max'],
                'avg': model_data['weight_average'],
                'std': model_data.get('weight_std', 'N/A'),
                'layers_exceeding': f"{model_data['layers_exceeding_threshold']}/{model_data['total_layers']}"
            })
            TH = model_data['threshold']
        except Exception as e:
            print(f"모델 '{model_name}' 데이터 로드 실패: {e}")

    fig = plt.figure(figsize=(30, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.01)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    ax1.set_ylabel('Probability Density', fontsize=40)

    for i, (model_name, bin_centers, hist_values, _) in enumerate(all_data):
        neg_indices = bin_centers < 0
        neg_centers = np.abs(bin_centers[neg_indices])
        neg_values = hist_values[neg_indices]
        if len(neg_centers) > 0:
            sort_idx = np.argsort(neg_centers)
            neg_centers = neg_centers[sort_idx]
            neg_values = neg_values[sort_idx]
            ax1.loglog(neg_centers, neg_values, color=colors[i % len(colors)], linewidth=12.0, label=model_name)

    for i, (model_name, bin_centers, hist_values, _) in enumerate(all_data):
        pos_indices = bin_centers > 0
        pos_centers = bin_centers[pos_indices]
        pos_values = hist_values[pos_indices]
        if len(pos_centers) > 0:
            ax2.loglog(pos_centers, pos_values, color=colors[i % len(colors)], linewidth=12.0)

    ax1.axvline(x=TH, color='red', linestyle='--', alpha=0.7, linewidth=12.0)
    th_line = ax2.axvline(x=TH, color='red', linestyle='--', alpha=0.7, linewidth=12.0)

    # ax1.grid(True, which='both', alpha=0.3)
    # ax2.grid(True, which='both', alpha=0.3)

    ax2.set_yticklabels([])

    y_min = min([min(data[2]) for data in all_data if len(data[2]) > 0]) / 10
    y_max = max([max(data[2]) for data in all_data if len(data[2]) > 0]) * 10
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)

    ax1.xaxis.set_major_locator(LogLocator(base=10.0))
    ax2.xaxis.set_major_locator(LogLocator(base=10.0))
    ax1.xaxis.set_minor_locator(LogLocator(base=10.0, subs='auto'))
    ax2.xaxis.set_minor_locator(LogLocator(base=10.0, subs='auto'))



    
    from matplotlib.ticker import FuncFormatter

    # 지수를 위 첨자 유니코드로 변환하는 함수
    def to_superscript(n):
        superscript_digits = {
            '-': '⁻',
            '0': '⁰',
            '1': '¹',
            '2': '²',
            '3': '³',
            '4': '⁴',
            '5': '⁵',
            '6': '⁶',
            '7': '⁷',
            '8': '⁸',
            '9': '⁹',
        }
        return ''.join(superscript_digits.get(c, c) for c in str(n))

    # negative x-axis tick formatter
    def negative_tick_formatter(x, pos):
        if x <= 0:
            return ""
        exponent = int(round(np.log10(x)))
        return f'−10{to_superscript(exponent)}'  # 유니코드 − 사용

    # 적용
    ax1.xaxis.set_major_formatter(FuncFormatter(negative_tick_formatter))


    ax1.tick_params(axis='both', which='major', labelsize=40)
    ax1.tick_params(axis='both', which='minor', labelsize=40)
    ax2.tick_params(axis='both', which='major', labelsize=40)
    ax2.tick_params(axis='both', which='minor', labelsize=40)

    ax1.invert_xaxis()
    # ax1.legend(fontsize=17, loc='upper left', bbox_to_anchor=(0.13, 1.0))
    plt.tight_layout()
    
    # 기존 라인들에서 handles, labels 추출
    handles, labels = ax1.get_legend_handles_labels()

    # TH = ±1.75 점선에 해당하는 라인 객체 추가
    handles.append(th_line)
    labels.append("TH = ±1.75")

    # 상단 중앙에 범례 표시 (추가된 항목 포함)
    fig.legend(
      handles,
      labels,
      loc='upper center',
      fontsize=36,
      ncol=len(labels),  # 항목 수와 동일한 열
      frameon=False,
      bbox_to_anchor=(0.5, 1.05),
      handlelength=1.8,      # 선 길이 (기본: 2.0)
      handletextpad=0.8,     # 선과 텍스트 간격 (기본: 0.8)
      columnspacing=1.0      # 열 간 간격 (기본: 2.0 → 줄이면 간격 좁아짐)
    )



    output_file = "model_weight_distributions_combined.png"
    
    plt.tight_layout()
    fig.text(0.5, -0.05, 'Weight Value (Log Scale)', ha='center', fontsize=40)

    plt.savefig(output_file, dpi=400, bbox_inches='tight')

    print("\n모델 통계 정보:")
    print(f"{'Model':<30} {'Min':<10} {'Max':<10} {'Avg':<12} {'Layers > TH'}")
    print("-" * 70)
    for stat in model_stats:
        print(f"{stat['name']:<30} {stat['min']:<10.4f} {stat['max']:<10.4f} {stat['avg']:<12.4e} {stat['layers_exceeding']}")
    print(f"\n그래프가 '{output_file}'에 저장되었습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='여러 모델의 가중치 분포를 통합된 로그-로그 스케일로 비교 시각화합니다.')
    parser.add_argument('models', nargs='*', help='모델 이름들 (선택사항: 기본값으로 고정된 4개 모델 사용)')
    args = parser.parse_args()
    if args.models:
        plot_weight_distributions(args.models)
    else:
        plot_weight_distributions()
