"""
Pruning 검증 스크립트
Magnitude와 Structured pruning이 제대로 적용되고 물리적으로 지워졌는지 확인
"""
import torch
import torch.nn as nn
from copy import deepcopy
import sys
import os

# 경로 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.test_pruning import (
    apply_magnitude_pruning,
    apply_structured_pruning,
    count_parameters
)
from src.utils import load_base_model, setup_device

def verify_pruning(model, pruned_model, pruning_rate, method_name):
    """Pruning이 제대로 적용되었는지 검증"""
    print(f"\n{'='*70}")
    print(f"=== {method_name} 검증 ===")
    print(f"{'='*70}")
    
    # 1. 파라미터 개수 확인
    old_params, old_trainable = count_parameters(model)
    new_params, new_trainable = count_parameters(pruned_model)
    reduction = (1 - new_params / old_params) * 100
    
    print(f"\n📊 파라미터 개수:")
    print(f"   원본: {old_params:,} 파라미터")
    print(f"   프루닝 후: {new_params:,} 파라미터")
    print(f"   감소율: {reduction:.2f}%")
    print(f"   예상 감소율: {pruning_rate*100:.0f}%")
    
    # 2. 물리적 구조 확인 (레이어 크기)
    print(f"\n🔍 물리적 구조 확인:")
    
    # Decoder 구조 확인
    original_decoder = model.decoder
    pruned_decoder = pruned_model.decoder
    
    # decoder_dim 확인
    if hasattr(original_decoder, 'decoder_dim') and hasattr(pruned_decoder, 'decoder_dim'):
        orig_dim = original_decoder.decoder_dim
        pruned_dim = pruned_decoder.decoder_dim
        dim_reduction = (1 - pruned_dim / orig_dim) * 100
        print(f"   decoder_dim: {orig_dim} → {pruned_dim} (감소: {dim_reduction:.1f}%)")
    
    # attention_dim 확인
    if hasattr(original_decoder, 'attention_dim') and hasattr(pruned_decoder, 'attention_dim'):
        orig_att = original_decoder.attention_dim
        pruned_att = pruned_decoder.attention_dim
        att_reduction = (1 - pruned_att / orig_att) * 100
        print(f"   attention_dim: {orig_att} → {pruned_att} (감소: {att_reduction:.1f}%)")
    
    # 3. 레이어별 가중치 크기 확인
    print(f"\n🔬 레이어별 가중치 크기:")
    
    layers_to_check = [
        ('decoder_att', 'decoder_att'),
        ('init_h', 'init_h'),
        ('fc', 'fc'),
        ('encoder_att', 'encoder_att'),
        ('decode_step', 'decode_step')
    ]
    
    for layer_name, attr_name in layers_to_check:
        if hasattr(original_decoder, attr_name) and hasattr(pruned_decoder, attr_name):
            orig_layer = getattr(original_decoder, attr_name)
            pruned_layer = getattr(pruned_decoder, attr_name)
            
            if isinstance(orig_layer, nn.Linear):
                orig_weight = orig_layer.weight.data
                pruned_weight = pruned_layer.weight.data
                print(f"   {layer_name}: {list(orig_weight.shape)} → {list(pruned_weight.shape)}")
            elif isinstance(orig_layer, nn.GRUCell):
                orig_weight_ih = orig_layer.weight_ih.data
                orig_weight_hh = orig_layer.weight_hh.data
                pruned_weight_ih = pruned_layer.weight_ih.data
                pruned_weight_hh = pruned_layer.weight_hh.data
                print(f"   {layer_name}.weight_ih: {list(orig_weight_ih.shape)} → {list(pruned_weight_ih.shape)}")
                print(f"   {layer_name}.weight_hh: {list(orig_weight_hh.shape)} → {list(pruned_weight_hh.shape)}")
    
    # 4. 실제 메모리 사용량 확인
    print(f"\n💾 메모리 사용량:")
    orig_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
    pruned_size = sum(p.numel() * p.element_size() for p in pruned_model.parameters()) / 1024 / 1024
    size_reduction = (1 - pruned_size / orig_size) * 100
    print(f"   원본: {orig_size:.2f} MB")
    print(f"   프루닝 후: {pruned_size:.2f} MB")
    print(f"   감소율: {size_reduction:.2f}%")
    
    # 5. 검증 결과
    print(f"\n✅ 검증 결과:")
    expected_reduction = pruning_rate * 100
    tolerance = 5.0  # 5% 허용 오차
    
    if abs(reduction - expected_reduction) <= tolerance:
        print(f"   ✅ 파라미터 감소율이 예상 범위 내입니다 ({reduction:.1f}% ≈ {expected_reduction:.0f}%)")
    else:
        print(f"   ⚠️ 파라미터 감소율이 예상과 다릅니다 ({reduction:.1f}% vs {expected_reduction:.0f}%)")
    
    if new_params < old_params:
        print(f"   ✅ 물리적 구조 변경 확인: 파라미터가 실제로 감소했습니다")
    else:
        print(f"   ❌ 물리적 구조 변경 실패: 파라미터가 감소하지 않았습니다")
    
    return {
        'old_params': old_params,
        'new_params': new_params,
        'reduction': reduction,
        'expected_reduction': expected_reduction,
        'physical_change': new_params < old_params
    }

def main():
    print("="*70)
    print("=== Pruning 검증 스크립트 ===")
    print("="*70)
    
    # 디바이스 설정
    device = setup_device()
    
    # 모델 로드
    print("\n모델 로드 중...")
    base_model, wm, rwm = load_base_model(device=device)
    base_model.eval()
    
    # 원본 파라미터 확인
    orig_params, _ = count_parameters(base_model)
    print(f"원본 모델 파라미터: {orig_params:,}")
    
    # 테스트할 프루닝 비율
    pruning_rate = 0.3  # 30%
    
    # 1. Magnitude Pruning 검증
    print("\n" + "="*70)
    print("Magnitude Pruning 적용 중...")
    print("="*70)
    try:
        magnitude_model = apply_magnitude_pruning(base_model, pruning_rate)
        magnitude_model.eval()
        magnitude_result = verify_pruning(
            base_model, magnitude_model, pruning_rate, "Magnitude Pruning"
        )
    except Exception as e:
        print(f"❌ Magnitude Pruning 실패: {e}")
        import traceback
        traceback.print_exc()
        magnitude_result = None
    
    # 2. Structured Pruning 검증
    print("\n" + "="*70)
    print("Structured Pruning 적용 중...")
    print("="*70)
    try:
        structured_model = apply_structured_pruning(base_model, pruning_rate)
        structured_model.eval()
        structured_result = verify_pruning(
            base_model, structured_model, pruning_rate, "Structured Pruning"
        )
    except Exception as e:
        print(f"❌ Structured Pruning 실패: {e}")
        import traceback
        traceback.print_exc()
        structured_result = None
    
    # 3. 비교 결과
    print("\n" + "="*70)
    print("=== 최종 비교 결과 ===")
    print("="*70)
    
    if magnitude_result and structured_result:
        print(f"\n{'항목':<30} {'Magnitude':<20} {'Structured':<20}")
        print("-"*70)
        print(f"{'파라미터 감소율':<30} {magnitude_result['reduction']:<20.2f} {structured_result['reduction']:<20.2f}")
        print(f"{'물리적 구조 변경':<30} {'✅' if magnitude_result['physical_change'] else '❌':<20} {'✅' if structured_result['physical_change'] else '❌':<20}")
        
        # 차이점 확인
        if abs(magnitude_result['reduction'] - structured_result['reduction']) < 1.0:
            print(f"\n⚠️ Magnitude와 Structured Pruning의 결과가 거의 동일합니다.")
            print(f"   이는 두 방법이 동일한 방식으로 구현되었기 때문일 수 있습니다.")
        else:
            print(f"\n✅ Magnitude와 Structured Pruning의 차이가 확인되었습니다.")
    
    print("\n" + "="*70)
    print("검증 완료")
    print("="*70)

if __name__ == "__main__":
    main()

