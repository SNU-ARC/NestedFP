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
            
            # 통계 정보 수집
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
            
            print(f"모델 '{model_name}' 데이터 로드 성공")
            
        except Exception as e:
            print(f"모델 '{model_name}' 데이터 로드 실패: {e}")
    
    # 더 통합된 느낌을 위해 특별한 레이아웃 사용
    fig = plt.figure(figsize=(16, 10))
    
    # gridspec을 사용하여 레이아웃 정의
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.01)
    
    # 왼쪽 (음수) 및 오른쪽 (양수) 서브플롯 생성
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    
    # y축 레이블 (왼쪽에만 표시)
    ax1.set_ylabel('Probability Density (log scale)', fontsize=14)
    
    # x축 레이블 (하단 중앙에 하나만 표시)
    fig.text(0.5, 0.02, 'Weight Value (log scale)', ha='center', fontsize=14)
    
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
            
            # 로그-로그 스케일로 플롯
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
            # 로그-로그 스케일로 플롯
            ax2.loglog(
                pos_centers, 
                pos_values, 
                color=colors[i % len(colors)], 
                linewidth=2.5, 
                label=model_name
            )
    
    # 임계값 표시
    ax1.axvline(x=TH, color='red', linestyle='--', alpha=0.7)
    ax2.axvline(x=TH, color='red', linestyle='--', alpha=0.7)
    
    # 텍스트로 음수/양수 구분 표시 (타이틀 대신)
    ax1.text(0.5, 0.95, 'Negative Weights', transform=ax1.transAxes, 
             ha='center', va='top', fontsize=12, alpha=0.7)
    ax2.text(0.5, 0.95, 'Positive Weights', transform=ax2.transAxes, 
             ha='center', va='top', fontsize=12, alpha=0.7)
    
    # 왼쪽 서브플롯 설정 (음수 부분)
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, which='both', alpha=0.3)
    
    # x축을 반대로 표시 (큰 값이 왼쪽으로)
    ax1.invert_xaxis()
    
    # 오른쪽 서브플롯 설정 (양수 부분)
    ax2.grid(True, which='both', alpha=0.3)
    
    # 오른쪽 y축 레이블 숨기기
    ax2.set_yticklabels([])
    
    # 오른쪽 서브플롯 범례는 상단 오른쪽에 배치
    ax2.legend(fontsize=10, loc='upper right')
    
    # y축 범위 동기화
    y_min = min([min(data[2]) for data in all_data if len(data[2]) > 0]) / 10
    y_max = max([max(data[2]) for data in all_data if len(data[2]) > 0]) * 10
    
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)
    
    # x축 범위 및 눈금 설정
    # 더 많은 눈금 표시 (로그 스케일 특성 강조)
    locmaj = LogLocator(base=10, numticks=10)
    ax1.xaxis.set_major_locator(locmaj)
    ax2.xaxis.set_major_locator(locmaj)
    
    # 출력용 눈금 형식 설정
    formatter = FormatStrFormatter('%.1e')
    ax1.xaxis.set_major_formatter(formatter)
    ax2.xaxis.set_major_formatter(formatter)
    
    # 전체 제목 추가
    plt.suptitle('Weight Distribution Comparison Across Models (Log-Log Scale)', fontsize=16)
    
    # 임계값 표시 텍스트 추가
    ax1.text(0.95, 0.05, f'-{TH}', transform=ax1.transAxes, 
             ha='right', va='bottom', color='red', fontsize=12)
    ax2.text(0.05, 0.05, f'+{TH}', transform=ax2.transAxes, 
             ha='left', va='bottom', color='red', fontsize=12)
    
    # 통계 정보 표 생성
    if model_stats:
        # 테이블 위치 및 크기 조정
        plt.subplots_adjust(bottom=0.25)
        
        # 테이블 데이터 준비
        table_data = []
        for stat in model_stats:
            row = [
                stat['name'],
                f"{stat['min']:.4f}",
                f"{stat['max']:.4f}",
                f"{stat['avg']:.4e}",
                f"{stat['layers_exceeding']}"
            ]
            table_data.append(row)
        
        # 테이블 생성
        table = plt.table(
            cellText=table_data,
            colLabels=['Model', 'Min', 'Max', 'Avg', 'Layers > TH'],
            loc='bottom',
            bbox=[0.15, -0.25, 0.7, 0.18]  # 테이블 위치 및 크기 조정
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
    
    # 레이아웃 조정
    plt.tight_layout(rect=[0, 0.12, 1, 0.95])  # 제목과 x축 레이블 공간 확보
    
    # 저장
    output_file = "model_weight_distributions_combined.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"통합된 로그-로그 스케일 그래프가 '{output_file}'에 저장되었습니다.")

if __name__ == "__main__":
    # 명령행 인자 파싱 (선택사항으로 변경)
    parser = argparse.ArgumentParser(description='여러 모델의 가중치 분포를 통합된 로그-로그 스케일로 비교 시각화합니다.')
    parser.add_argument('models', nargs='*', help='모델 이름들 (선택사항: 기본값으로 고정된 4개 모델 사용)')
    args = parser.parse_args()
    
    # 명령행 인자가 있으면 사용, 없으면 기본값 사용
    if args.models:
        plot_weight_distributions(args.models)
    else:
        print("기본 모델 목록 사용:")
        print("- Llama-3.1-8B")
        print("- Mistral-Nemo-Base-2407")
        print("- Mistral-Small-24B-Base-2501")
        print("- phi-4")
        plot_weight_distributions()