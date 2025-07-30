import torch
import json
import numpy as np
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
from tqdm import tqdm

def classify_gemm_type(layer_name):
    """
    논문 기준에 따른 GEMM 타입 분류
    GEMM1: QKV projections (q_proj, k_proj, v_proj)
    GEMM2: output projections (o_proj)
    GEMM3: MLP gate/up projections (gate_proj, up_proj)
    GEMM4: MLP down projections (down_proj)
    """
    layer_name_lower = layer_name.lower()
    
    if any(proj in layer_name_lower for proj in ['q_proj', 'k_proj', 'v_proj']):
        return 'GEMM1'
    elif 'o_proj' in layer_name_lower:
        return 'GEMM2'
    elif any(proj in layer_name_lower for proj in ['gate_proj', 'up_proj']):
        return 'GEMM3'
    elif 'down_proj' in layer_name_lower:
        return 'GEMM4'
    else:
        return 'Other'

def analyze_model_weights_gpu(model_path, threshold=1.75, output_prefix=None):
    """
    GPU를 활용한 고속 모델 가중치 분석
    """
    if output_prefix is None:
        output_prefix = os.path.basename(model_path).replace("/", "_")
    
    print(f"모델 로딩 중: {model_path}")
    print(f"사용 가능한 GPU: {torch.cuda.device_count()}개")
    
    try:
        # GPU에서 모델 로딩 (다중 GPU 활용)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",  # 자동으로 여러 GPU에 분산
            trust_remote_code=True
        )
        
        print(f"모델 로딩 완료. GPU에서 가중치 분석 시작...")
        
        # 전체 가중치 수집을 위한 리스트 (CPU에서 관리)
        all_weights_cpu = []
        layer_info = []
        
        # GEMM별 통계 초기화
        gemm_stats = {
            'GEMM1': {'total': 0, 'applicable': 0, 'layers': []},
            'GEMM2': {'total': 0, 'applicable': 0, 'layers': []},
            'GEMM3': {'total': 0, 'applicable': 0, 'layers': []},
            'GEMM4': {'total': 0, 'applicable': 0, 'layers': []},
            'Other': {'total': 0, 'applicable': 0, 'layers': []}
        }
        
        total_layers = 0
        layers_exceeding = 0
        threshold_tensor = torch.tensor(threshold, dtype=torch.float16)
        
        # 레이어별 분석 (GPU에서 빠른 연산)
        for name, param in tqdm(model.named_parameters(), desc="GPU에서 레이어 분석"):
            if param.requires_grad and len(param.shape) > 1:  # 2D 이상의 가중치만 분석
                total_layers += 1
                
                # GPU에서 직접 통계 계산 (훨씬 빠름!)
                with torch.no_grad():
                    abs_param = torch.abs(param)
                    layer_max = torch.max(abs_param).item()
                    layer_min = torch.min(abs_param).item()
                    layer_mean = torch.mean(abs_param).item()
                    layer_std = torch.std(abs_param).item()
                
                # NestedFP 적용 가능성 (GPU에서 비교)
                is_applicable = layer_max <= threshold
                if not is_applicable:
                    layers_exceeding += 1
                
                # GEMM 타입 분류
                gemm_type = classify_gemm_type(name)
                gemm_stats[gemm_type]['total'] += 1
                if is_applicable:
                    gemm_stats[gemm_type]['applicable'] += 1
                
                layer_detail = {
                    'name': name,
                    'gemm_type': gemm_type,
                    'shape': list(param.shape),
                    'max_abs': float(layer_max),
                    'min_abs': float(layer_min),
                    'mean_abs': float(layer_mean),
                    'std_abs': float(layer_std),
                    'is_applicable': is_applicable,
                    'num_params': param.numel(),
                    'device': str(param.device)  # 어느 GPU에 있는지 확인
                }
                
                layer_info.append(layer_detail)
                gemm_stats[gemm_type]['layers'].append(layer_detail)
                
                status = "✓" if is_applicable else "✗"
                print(f"{status} {gemm_type:6} {name}: shape={param.shape}, max_abs={layer_max:.4f}, device={param.device}")
                
                # 히스토그램용 데이터는 CPU로 이동 (메모리 관리)
                if total_layers % 10 == 0:  # 10개 레이어마다 수집
                    weights_sample = param.detach().cpu().numpy().flatten()
                    # 샘플링해서 메모리 절약
                    if len(weights_sample) > 1000000:  # 100만개 이상이면 샘플링
                        indices = np.random.choice(len(weights_sample), 1000000, replace=False)
                        weights_sample = weights_sample[indices]
                    all_weights_cpu.extend(weights_sample)
        
        print("GPU 분석 완료, 히스토그램 생성 중...")
        
        # 전체 가중치 통계 계산 (샘플 기반)
        all_weights = np.array(all_weights_cpu)
        
        # 히스토그램 생성 (로그 스케일)
        non_zero_weights = all_weights[all_weights != 0]
        
        if len(non_zero_weights) > 0:
            # 양수와 음수 분리
            pos_weights = non_zero_weights[non_zero_weights > 0]
            neg_weights = non_zero_weights[non_zero_weights < 0]
            
            # 로그 스케일 빈 생성
            min_log = np.log10(np.min(np.abs(non_zero_weights)))
            max_log = np.log10(np.max(np.abs(non_zero_weights)))
            bins = np.logspace(min_log, max_log, 100)
            
            # 양수 히스토그램
            hist_pos, _ = np.histogram(pos_weights, bins=bins, density=True)
            # 음수 히스토그램 (절대값 사용)
            hist_neg, _ = np.histogram(np.abs(neg_weights), bins=bins, density=True)
            
            # 빈 중심값 계산
            bin_centers_pos = bins[:-1] + (bins[1:] - bins[:-1]) / 2
            bin_centers_neg = -(bins[:-1] + (bins[1:] - bins[:-1]) / 2)
            
            # 전체 히스토그램 결합
            bin_centers = np.concatenate([bin_centers_neg[::-1], bin_centers_pos])
            hist_values = np.concatenate([hist_neg[::-1], hist_pos])
        else:
            bin_centers = []
            hist_values = []
        
        # 전체 파라미터 수 계산 (GPU에서)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # GEMM별 요약 통계 계산
        gemm_summary = {}
        for gemm_type, stats in gemm_stats.items():
            if stats['total'] > 0:
                applicability_rate = (stats['applicable'] / stats['total']) * 100
                gemm_summary[gemm_type] = {
                    'applicable_layers': stats['applicable'],
                    'total_layers': stats['total'],
                    'applicability_rate': round(applicability_rate, 1),
                    'format_string': f"{stats['applicable']}/{stats['total']} ({applicability_rate:.1f}%)"
                }
        
        # 전체 통계 정보
        total_applicable = total_layers - layers_exceeding
        overall_applicability = (total_applicable / total_layers) * 100 if total_layers > 0 else 0
        
        stats = {
            'model_path': model_path,
            'model_name': output_prefix,
            'threshold': threshold,
            'analysis_method': 'GPU_accelerated',
            'gpu_count': torch.cuda.device_count(),
            'analysis_summary': {
                'total_layers': total_layers,
                'applicable_layers': total_applicable,
                'layers_exceeding_threshold': layers_exceeding,
                'overall_applicability_rate': round(overall_applicability, 1),
                'format_string': f"{total_applicable}/{total_layers} ({overall_applicability:.1f}%)"
            },
            'gemm_analysis': gemm_summary,
            'weight_statistics': {
                'min': float(np.min(all_weights)) if len(all_weights) > 0 else 0,
                'max': float(np.max(all_weights)) if len(all_weights) > 0 else 0,
                'mean': float(np.mean(all_weights)) if len(all_weights) > 0 else 0,
                'std': float(np.std(all_weights)) if len(all_weights) > 0 else 0,
                'total_parameters': total_params,
                'sampled_parameters': len(all_weights)
            },
            'histogram': {
                'bin_centers': bin_centers.tolist(),
                'hist_values': hist_values.tolist()
            },
            'layer_details': layer_info,
            'gemm_details': {k: v['layers'] for k, v in gemm_stats.items() if v['total'] > 0}
        }
        
        # JSON 파일로 저장
        output_file = f"{output_prefix}_nestedfp_analysis_gpu.json"
        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # 결과 출력
        print(f"\n" + "="*70)
        print(f"NestedFP 적용 가능성 분석 결과 (GPU 가속)")
        print(f"="*70)
        print(f"모델명: {output_prefix}")
        print(f"임계값: {threshold}")
        print(f"사용된 GPU: {torch.cuda.device_count()}개")
        print(f"전체 파라미터 수: {stats['weight_statistics']['total_parameters']:,}")
        print()
        
        print("GEMM별 적용 가능성:")
        print("-" * 50)
        for gemm_type in ['GEMM1', 'GEMM2', 'GEMM3', 'GEMM4']:
            if gemm_type in gemm_summary:
                info = gemm_summary[gemm_type]
                print(f"{gemm_type:6}: {info['format_string']}")
            else:
                print(f"{gemm_type:6}: 해당 레이어 없음")
        
        if 'Other' in gemm_summary:
            info = gemm_summary['Other']
            print(f"{'Other':6}: {info['format_string']}")
        
        print("-" * 50)
        summary = stats['analysis_summary']
        print(f"{'Total':6}: {summary['format_string']}")
        print()
        
        print("GEMM 타입 설명:")
        print("- GEMM1: QKV projections (q_proj, k_proj, v_proj)")
        print("- GEMM2: Output projections (o_proj)")  
        print("- GEMM3: MLP gate/up projections (gate_proj, up_proj)")
        print("- GEMM4: MLP down projections (down_proj)")
        print()
        print(f"결과가 '{output_file}'에 저장되었습니다.")
        
        # GPU 메모리 정리
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
        return output_file
        
    except Exception as e:
        print(f"GPU 모델 분석 중 오류 발생: {e}")
        # GPU 메모리 정리
        torch.cuda.empty_cache()
        return None

def analyze_deepseek_model_gpu():
    """DeepSeek-R1-Distill-Llama-70B 모델 GPU 분석"""
    model_path = "/disk2/models/DeepSeek-R1-Distill-Llama-70B"
    
    # 모델 존재 여부 확인
    if not os.path.exists(model_path):
        print(f"모델 경로를 찾을 수 없습니다: {model_path}")
        return None
    
    # GPU 확인
    if not torch.cuda.is_available():
        print("CUDA를 사용할 수 없습니다. CPU 버전을 사용하세요.")
        return None
    
    print(f"DeepSeek-R1-Distill-Llama-70B 모델의 NestedFP 분석 (GPU 가속)")
    print(f"GPU 개수: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 분석 실행
    result_file = analyze_model_weights_gpu(
        model_path=model_path,
        threshold=1.75,
        output_prefix="DeepSeek-R1-Distill-Llama-70B"
    )
    
    return result_file

if __name__ == "__main__":
    # DeepSeek 모델 GPU 분석
    result = analyze_deepseek_model_gpu()
    
    if result:
        print(f"\nGPU 분석 완료! 다음 명령으로 시각화할 수 있습니다:")
        print(f"python plot_weight_distributions.py DeepSeek-R1-Distill-Llama-70B")