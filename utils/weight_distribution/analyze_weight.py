import torch
import json
import numpy as np
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
from tqdm import tqdm

def analyze_model_weights(model_path, threshold=1.75, output_prefix=None):
    """
    모델의 가중치를 분석하고 통계를 JSON 파일로 저장
    """
    if output_prefix is None:
        output_prefix = os.path.basename(model_path).replace("/", "_")
    
    print(f"모델 로딩 중: {model_path}")
    
    try:
        # 모델 로딩 (CPU에서 로딩하여 메모리 절약)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # 메모리 절약을 위해 float16 사용
            device_map="cpu",  # CPU에서 분석
            trust_remote_code=True
        )
        
        print(f"모델 로딩 완료. 가중치 분석 시작...")
        
        # 모든 가중치 수집
        all_weights = []
        layer_info = []
        layers_exceeding = 0
        total_layers = 0
        
        # 레이어별 분석
        for name, param in tqdm(model.named_parameters(), desc="레이어 분석"):
            if param.requires_grad and len(param.shape) > 1:  # 2D 이상의 가중치만 분석
                total_layers += 1
                
                # 가중치를 numpy 배열로 변환
                weights = param.detach().cpu().numpy().flatten()
                all_weights.extend(weights)
                
                # 레이어별 통계
                layer_max = np.max(np.abs(weights))
                layer_min = np.min(np.abs(weights))
                layer_mean = np.mean(np.abs(weights))
                layer_std = np.std(np.abs(weights))
                
                # 임계값 초과 여부
                exceeds_threshold = layer_max > threshold
                if exceeds_threshold:
                    layers_exceeding += 1
                
                layer_info.append({
                    'name': name,
                    'shape': list(param.shape),
                    'max_abs': float(layer_max),
                    'min_abs': float(layer_min),
                    'mean_abs': float(layer_mean),
                    'std_abs': float(layer_std),
                    'exceeds_threshold': exceeds_threshold,
                    'num_params': param.numel()
                })
                
                print(f"레이어 {name}: shape={param.shape}, max_abs={layer_max:.4f}, exceeds_th={exceeds_threshold}")
        
        # 전체 가중치 통계 계산
        all_weights = np.array(all_weights)
        
        # 히스토그램 생성 (로그 스케일)
        # 0이 아닌 가중치만 사용
        non_zero_weights = all_weights[all_weights != 0]
        
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
        
        # 통계 정보
        stats = {
            'model_path': model_path,
            'model_name': output_prefix,
            'threshold': threshold,
            'total_layers': total_layers,
            'layers_exceeding_threshold': layers_exceeding,
            'weight_min': float(np.min(all_weights)),
            'weight_max': float(np.max(all_weights)),
            'weight_average': float(np.mean(all_weights)),
            'weight_std': float(np.std(all_weights)),
            'total_parameters': len(all_weights),
            'histogram': {
                'bin_centers': bin_centers.tolist(),
                'hist_values': hist_values.tolist()
            },
            'layer_details': layer_info
        }
        
        # JSON 파일로 저장
        output_file = f"{output_prefix}_stats.json"
        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n=== 분석 결과 ===")
        print(f"모델명: {output_prefix}")
        print(f"전체 레이어 수: {total_layers}")
        print(f"임계값({threshold}) 초과 레이어: {layers_exceeding}")
        print(f"NestedFP 적용 가능 레이어: {total_layers - layers_exceeding}")
        print(f"가중치 범위: {stats['weight_min']:.6f} ~ {stats['weight_max']:.6f}")
        print(f"가중치 평균: {stats['weight_average']:.6e}")
        print(f"전체 파라미터 수: {stats['total_parameters']:,}")
        print(f"결과가 '{output_file}'에 저장되었습니다.")
        
        # 메모리 정리
        del model
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return output_file
        
    except Exception as e:
        print(f"모델 분석 중 오류 발생: {e}")
        return None

def analyze_deepseek_model():
    """DeepSeek-R1-Distill-Llama-70B 모델 분석"""
    model_path = "/disk2/models/DeepSeek-R1-Distill-Llama-70B"
    
    # 모델 존재 여부 확인
    if not os.path.exists(model_path):
        print(f"모델 경로를 찾을 수 없습니다: {model_path}")
        return None
    
    print("DeepSeek-R1-Distill-Llama-70B 모델 분석을 시작합니다...")
    print("주의: 70B 모델은 분석에 상당한 시간과 메모리가 필요할 수 있습니다.")
    
    # 분석 실행
    result_file = analyze_model_weights(
        model_path=model_path,
        threshold=1.75,
        output_prefix="DeepSeek-R1-Distill-Llama-70B"
    )
    
    return result_file

if __name__ == "__main__":
    # DeepSeek 모델 분석
    result = analyze_deepseek_model()
    
    if result:
        print(f"\n분석 완료! 이제 다음 명령으로 시각화할 수 있습니다:")
        print(f"python plot_weight_distributions.py DeepSeek-R1-Distill-Llama-70B")