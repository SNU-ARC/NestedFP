import json
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import argparse
from matplotlib.ticker import LogLocator, FormatStrFormatter
import matplotlib.gridspec as gridspec

def load_json_robust(file_path):
    """문제가 있는 JSON 파일을 로드하는 강건한 함수"""
    try:
        # 일반적인 방법으로 시도
        with open(file_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류: {e}")
        print("수동 처리 시도 중...")
        
        # 파일 내용 읽기
        with open(file_path, 'r') as f:
            content = f.read()
        
        # 일반적인 JSON 문제 수정 시도
        # NaN 값 "null"로 대체
        content = re.sub(r':\s*NaN', ': null', content)
        
        # 후행 쉼표 제거
        content = re.sub(r',(\s*[\]}])', r'\1', content)
        
        # 따옴표 수정 시도
        content = content.replace('\'', '"')
        
        # 다시 파싱 시도
        try:
            return json.loads(content)
        except json.JSONDecodeError as e2:
            print(f"수동 처리 후에도 파싱 실패: {e2}")
            raise e2

def plot_weight_distributions(model_names=None):
    # 고정된 모델 목록 설정 (기본값)
    if model_names is None:
        model_names = [
            'Llama-3.1-8B',
            'Mistral-Nemo-Base-2407',
            'Mistral-Small-24B-Base-2501',
            'phi-4'
        ]
    
    # 그래프 스타일 설정
    plt.style.use('seaborn-v0_8')
    
    # 임계값 (모든 모델이 같은 임계값을 사용한다고 가정)
    TH = 1.75
    
    # 모델별 색상 정의
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # 각 모델에 대한 통계를 담을 리스트
    model_stats = []
    
    # 데이터를 담을 리스트
    all_data = []
    
    # 각 모델 데이터 로드
    for i, model_name in enumerate(model_names):
        json_file = f"{model_name}_stats.json"
        
        try:
            # 로버스트한 방법으로 JSON 로드
            model_data = load_json_robust(json_file)
            
            # 히스토그램 데이터 추출
            bin_centers = np.array(model_data['histogram']['bin_centers'])
            hist_values = np.array(model_data['histogram']['hist_values'])
            
            # 0이 아닌 데이터만 유지
            valid_indices = np.where(hist_values > 0)[0]
            bin_centers = bin_centers[valid_indices]
            hist_values = hist_values[valid_indices]
            
            # 데이터 저장
            all_data.append((model_name, bin_centers, hist_values, model_data))
            
            # 통계 정보 수집 (터미널 출력용)
            model_stats.append({
                'name': model_name,
                'min': model_data['weight_min'],
                'max': model_data['weight_max'],
                'avg': model_data['weight_average'],
                'std': model_data.get('weight_std', 'N/A'),
                'layers_exceeding': f"{model_data['layers_exceeding_threshold']}/{model_data['total_layers']}"
            })
            
            # 임계값 업데이트 (모든 모델이 같은 임계값을 사용할 것으로 가정)
            TH = model_data['threshold']
            
        except Exception as e:
            print(f"모델 '{model_name}' 데이터 로드 실패: {e}")
    
    # 더 통합된 느낌을 위해 특별한 레이아웃 사용
    fig = plt.figure(figsize=(20, 6))
    
    # gridspec을 사용하여 레이아웃 정의
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.01)
    
    # 왼쪽 (음수) 및 오른쪽 (양수) 서브플롯 생성
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    
    # y축 레이블 (왼쪽에만 표시)
    ax1.set_ylabel('Probability Density', fontsize=28)
    
    # x축 레이블 (하단 중앙에 하나만 표시)
    # fig.text(0.5, 0.08, 'Weight Value', ha='center', fontsize=28)
    
    # 각 모델 플롯 - 음수 부분 (왼쪽 서브플롯)
    for i, (model_name, bin_centers, hist_values, _) in enumerate(all_data):
        # 음수 데이터만 필터링
        neg_indices = bin_centers < 0
        neg_centers = np.abs(bin_centers[neg_indices])  # 절대값으로 변환
        neg_values = hist_values[neg_indices]
        
        if len(neg_centers) > 0:
            # 데이터 정렬 (음수는 오른쪽에서 왼쪽으로 표시)
            sort_idx = np.argsort(-neg_centers)
            neg_centers = neg_centers[sort_idx]
            neg_values = neg_values[sort_idx]
            
            # 로그-로그 스케일로 플롯 (label 유지)
            ax1.loglog(
                neg_centers, 
                neg_values, 
                color=colors[i % len(colors)], 
                linewidth=2.5, 
                label=model_name
            )
    
    # 각 모델 플롯 - 양수 부분 (오른쪽 서브플롯)
    for i, (model_name, bin_centers, hist_values, _) in enumerate(all_data):
        # 양수 데이터만 필터링
        pos_indices = bin_centers > 0
        pos_centers = bin_centers[pos_indices]
        pos_values = hist_values[pos_indices]
        
        if len(pos_centers) > 0:
            # 로그-로그 스케일로 플롯 (label 제거)
            ax2.loglog(
                pos_centers, 
                pos_values, 
                color=colors[i % len(colors)], 
                linewidth=2.5
            )
    
    # 임계값 표시
    ax1.axvline(x=TH, color='red', linestyle='--', alpha=0.7)
    ax2.axvline(x=TH, color='red', linestyle='--', alpha=0.7)
    
    # 왼쪽 서브플롯 설정 (음수 부분) - 범례 추가
    ax1.grid(True, which='both', alpha=0.3)
    ax1.legend(fontsize=17, loc='upper left', bbox_to_anchor=(0.13, 1.0))
    
    # x축을 반대로 표시 (큰 값이 왼쪽으로)
    ax1.invert_xaxis()
    
    # 오른쪽 서브플롯 설정 (양수 부분)
    ax2.grid(True, which='both', alpha=0.3)
    
    # 오른쪽 y축 레이블 숨기기
    ax2.set_yticklabels([])
    
    # y축 범위 동기화
    y_min = min([min(data[2]) for data in all_data if len(data[2]) > 0]) / 10
    y_max = max([max(data[2]) for data in all_data if len(data[2]) > 0]) * 10
    
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)
    
    # x축 범위 및 눈금 설정 - 추가 눈금 포함
    # 0.05 등 특정 값들을 명시적으로 추가
    
    # 주요 눈금 위치 설정 (표준 로그 스케일 + 추가 눈금)
    # major_ticks = [0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    major_ticks = [0.02, 0.1,1.75]
    
    # 각 축에 수동으로 주요 눈금 설정
    ax1.set_xticks(major_ticks)
    ax2.set_xticks(major_ticks)
    
    # 더 세밀한 눈금들을 위한 minor locator
    minor_ticks = []
    # 0.01-0.1 구간
    for i in range(4, 10):  # 0.04, 0.06, 0.07, 0.08, 0.09
        if i * 0.01 not in major_ticks:
            minor_ticks.append(i * 0.01)
    # 0.1-1.0 구간
    for i in range(4, 10):  # 0.4, 0.6, 0.7, 0.8, 0.9
        if i * 0.1 not in major_ticks:
            minor_ticks.append(i * 0.1)
    
    ax1.set_xticks(minor_ticks, minor=True)
    ax2.set_xticks(minor_ticks, minor=True)
    
     # 출력용 눈금 형식 설정 - 1.75 등 특정 값이 올바르게 표시되도록 개선
    def custom_formatter(x, pos):
        # 특정 값들을 명시적으로 처리
        if abs(x - 1.75) < 0.001:  # 1.75 근처
            return '1.75'
        elif abs(x - 0.5) < 0.001:  # 0.5 근처
            return '0.5'
        elif x >= 1:
            return f'{x:.1f}'
        else:
            return f'{x:.2f}'
    
    from matplotlib.ticker import FuncFormatter
    ax1.xaxis.set_major_formatter(FuncFormatter(custom_formatter))
    ax2.xaxis.set_major_formatter(FuncFormatter(custom_formatter))
    
    # tick 폰트 크기 설정
    ax1.tick_params(axis='both', which='major', labelsize=26)
    ax1.tick_params(axis='both', which='minor', labelsize=26)
    ax2.tick_params(axis='both', which='major', labelsize=26)
    ax2.tick_params(axis='both', which='minor', labelsize=26)
    
    # 레이아웃 조정
    plt.tight_layout()
    
    # 저장
    output_file = "model_weight_distributions_combined.png"
    plt.savefig(output_file, dpi=400, bbox_inches='tight')
    
    # 터미널에 통계 정보 출력
    print("\n모델 통계 정보:")
    print(f"{'Model':<30} {'Min':<10} {'Max':<10} {'Avg':<12} {'Layers > TH'}")
    print("-" * 70)
    for stat in model_stats:
        print(f"{stat['name']:<30} {stat['min']:<10.4f} {stat['max']:<10.4f} {stat['avg']:<12.4e} {stat['layers_exceeding']}")
    print(f"\n그래프가 '{output_file}'에 저장되었습니다.")

if __name__ == "__main__":
    # 명령행 인자 파싱 (선택사항으로 변경)
    parser = argparse.ArgumentParser(description='여러 모델의 가중치 분포를 통합된 로그-로그 스케일로 비교 시각화합니다.')
    parser.add_argument('models', nargs='*', help='모델 이름들 (선택사항: 기본값으로 고정된 4개 모델 사용)')
    args = parser.parse_args()
    
    # 명령행 인자가 있으면 사용, 없으면 기본값 사용
    if args.models:
        plot_weight_distributions(args.models)
    else:
        plot_weight_distributions()