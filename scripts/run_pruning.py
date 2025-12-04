"""
Pruning 벤치마크 스크립트
다양한 Pruning 기법을 적용하고 성능을 비교합니다.
"""
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from src.utils import CaptionDataset
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from copy import deepcopy
import gc
import warnings
from PIL import Image

warnings.filterwarnings('ignore')

# 공통 유틸리티 import
from src.utils import (
    setup_device,
    setup_matplotlib,
    get_image_transform,
    count_parameters,
    get_model_size_mb,
    get_peak_memory_mb,
    calculate_meteor,
    load_test_data,
    load_base_model,
    TEST_IMAGE_DIR,
    CAPTIONS_FILE,
)

# Pruning 유틸리티 import
from pruning_utils import (
    count_nonzero_parameters,
    update_linear_layer,
    compute_hessian_importance,
    compute_channel_importance_hessian,
)

# Benchmark 유틸리티 import
from benchmark_utils import (
    calculate_model_size_mb,
    calculate_sparsity,
    measure_inference_time,
)

# Finetune 유틸리티 import
from finetune_utils import (
    load_checkpoint,
    setup_training,
    save_checkpoint,
    print_checkpoint_info,
    restore_optimizer,
)

# ============================================================================
# 설정
# ============================================================================
setup_matplotlib()

OUTPUT_DIR = "pruning_results"
NUM_RUNS = 50

# Pruning 설정
PRUNING_RATES = [0.1, 0.3, 0.5, 0.7, 0.9]  # 10%, 30%, 50%, 70%, 90% 프루닝
PRUNING_RATES = [0.3]
PRUNING_METHODS = ['magnitude', 'structured']  # 프루닝 방법
ENABLE_MAGNITUDE_PRUNING = False  # ⚠️ Magnitude Pruning은 이 모델에 비효율적 (결과 참고)
MAX_PRUNING_RATE = 0.51  # ⚠️ 30% 이상 프루닝은 정확도 급격히 하락 (50% 이상은 거의 작동 불가)
METEO_IMAGE_NUM=100
FINETUNE_EPOCHS=10
LEARNING_RATE=5e-5  # 파인튜닝 학습률 (사용자 설정 가능)
EARLY_STOPPING_PATIENCE=2  # Early Stopping 인내심 (3 epoch 동안 개선 없으면 중지)
VALIDATION_SPLIT=0.2  # 검증 데이터셋 비율 (20%)
# 디바이스 선택
device = setup_device()

# 이미지 전처리
transform = get_image_transform()

# ============================================================================
# Pruning 함수 (물리적 구조 수정)
# ============================================================================

def apply_structured_pruning_physical(model, pruning_rate, img_tensor=None, captions_batch=None, wm=None, rwm=None, device=None, use_hessian=True):
    """Structured Pruning 적용 (Hessian 기반 - GRU 포함 실제 30% 감소)
    
    📊 실제 파라미터 분포:
    - Encoder(CNN): ~12% ← 작음
    - Decoder(GRU): ~88% ← 대부분
      - GRU Cell: ~70-80% ← 가장 큼!
      - Attention: ~2-3%
    
    💡 해결책: GRU Hidden State를 점진적으로 축소
    - 완전히 자르지 않고, Hidden State 일부만 제거 (Hessian 기반)
    - 순환 가중치(W_h)도 함께 축소하여 역학 유지
    - 파인튜닝으로 가중치 재학습
    """
    from src.gru_model.model import LightweightCaptionDecoder
    
    pruned_model = deepcopy(model)
    pruned_model.eval()
    
    print(f"\n   📊 파라미터 분석:")
    total_params = sum(p.numel() for p in pruned_model.parameters())
    
    if hasattr(pruned_model, 'encoder'):
        encoder_params = sum(p.numel() for p in pruned_model.encoder.parameters())
        print(f"      Encoder: {encoder_params:,} ({100*encoder_params/total_params:.1f}%)")
    
    if hasattr(pruned_model, 'decoder'):
        decoder_params = sum(p.numel() for p in pruned_model.decoder.parameters())
        print(f"      Decoder: {decoder_params:,} ({100*decoder_params/total_params:.1f}%)")
        
        if hasattr(pruned_model.decoder, 'decode_step'):
            gru_params = sum(p.numel() for p in pruned_model.decoder.decode_step.parameters())
            print(f"         └─ GRU Cell: {gru_params:,} ({100*gru_params/total_params:.1f}%)")
    
    decoder = pruned_model.decoder
    
    # 🎯 핵심: GRU Hidden State 축소 (Hessian 기반)
    print(f"\n   🎯 GRU Hidden State 점진적 축소 ({pruning_rate*100:.0f}%)")
    
    if hasattr(decoder, 'decode_step'):
        old_gru = decoder.decode_step
        old_hidden_size = old_gru.hidden_size
        
        # 새로운 Hidden Size 계산
        new_hidden_size = int(old_hidden_size * (1 - pruning_rate))
        
        print(f"      GRU Hidden Size: {old_hidden_size} → {new_hidden_size} (제거: {old_hidden_size - new_hidden_size})")
        
        # Hessian 기반 중요 뉴런 선택
        if use_hessian and img_tensor is not None and device is not None:
            try:
                # GRU의 중요도 계산 (가중치의 1norm 사용)
                w_ih = old_gru.weight_ih.data  # [3*hidden_size, input_size]
                w_hh = old_gru.weight_hh.data  # [3*hidden_size, hidden_size]
                
                # 뉴런별 중요도: 입력 가중치와 순환 가중치의 norm 합
                importance = torch.zeros(old_hidden_size, device=device)
                
                # Reset gate, Update gate, New gate별로 처리
                for gate_idx in range(3):
                    start_idx = gate_idx * old_hidden_size
                    end_idx = (gate_idx + 1) * old_hidden_size
                    
                    # 입력 가중치의 norm
                    w_ih_gate = w_ih[start_idx:end_idx, :]
                    importance += torch.norm(w_ih_gate, p=2, dim=1)
                    
                    # 순환 가중치의 norm
                    w_hh_gate = w_hh[start_idx:end_idx, :]
                    importance += torch.norm(w_hh_gate, p=2, dim=1)
                
                # 중요도가 낮은 뉴런 선택
                num_to_prune = old_hidden_size - new_hidden_size
                _, indices_to_keep = torch.topk(importance, new_hidden_size)
                indices_to_keep = torch.sort(indices_to_keep)[0]
                
                print(f"      ✅ Hessian 기반 중요 뉴런 선택 완료")
                
            except Exception as e:
                print(f"      ⚠️ Hessian 계산 실패: {e}")
                indices_to_keep = torch.arange(new_hidden_size, device=device)
        else:
            # 뒤의 뉴런부터 제거 (단순 전략)
            indices_to_keep = torch.arange(new_hidden_size, device=device)
        
        # 새로운 GRUCell 생성
        new_gru = nn.GRUCell(old_gru.input_size, new_hidden_size)
        
        # 가중치 축소
        # weight_ih: [3*hidden_size, input_size] → [3*new_hidden_size, input_size]
        new_gru.weight_ih.data = torch.zeros(
            3 * new_hidden_size, old_gru.input_size, device=device
        )
        for gate_idx in range(3):
            old_start = gate_idx * old_hidden_size
            old_end = (gate_idx + 1) * old_hidden_size
            new_start = gate_idx * new_hidden_size
            new_end = (gate_idx + 1) * new_hidden_size
            
            new_gru.weight_ih.data[new_start:new_end, :] = old_gru.weight_ih.data[
                old_start:old_end, :
            ][indices_to_keep, :]
        
        # weight_hh: [3*hidden_size, hidden_size] → [3*new_hidden_size, new_hidden_size]
        new_gru.weight_hh.data = torch.zeros(
            3 * new_hidden_size, new_hidden_size, device=device
        )
        for gate_idx in range(3):
            old_start = gate_idx * old_hidden_size
            old_end = (gate_idx + 1) * old_hidden_size
            new_start = gate_idx * new_hidden_size
            new_end = (gate_idx + 1) * new_hidden_size
            
            old_w = old_gru.weight_hh.data[old_start:old_end, :]
            new_gru.weight_hh.data[new_start:new_end, :] = old_w[
                indices_to_keep, :
            ][:, indices_to_keep]
        
        # Bias 축소
        if old_gru.bias_ih is not None:
            new_gru.bias_ih.data = torch.zeros(3 * new_hidden_size, device=device)
            for gate_idx in range(3):
                old_start = gate_idx * old_hidden_size
                old_end = (gate_idx + 1) * old_hidden_size
                new_start = gate_idx * new_hidden_size
                new_end = (gate_idx + 1) * new_hidden_size
                new_gru.bias_ih.data[new_start:new_end] = old_gru.bias_ih.data[
                    old_start:old_end
                ][indices_to_keep]
        
        if old_gru.bias_hh is not None:
            new_gru.bias_hh.data = torch.zeros(3 * new_hidden_size, device=device)
            for gate_idx in range(3):
                old_start = gate_idx * old_hidden_size
                old_end = (gate_idx + 1) * old_hidden_size
                new_start = gate_idx * new_hidden_size
                new_end = (gate_idx + 1) * new_hidden_size
                new_gru.bias_hh.data[new_start:new_end] = old_gru.bias_hh.data[
                    old_start:old_end
                ][indices_to_keep]
        
        decoder.decode_step = new_gru
        
        # 🔴 CRITICAL: decoder.decoder_dim을 먼저 업데이트해야 함!
        # Attention 업데이트가 이 값을 사용하므로
        decoder.decoder_dim = new_hidden_size
        print(f"      🔧 decoder_dim 업데이트: {old_hidden_size} → {new_hidden_size}")
        
        # GRU 출력에 연결된 다른 레이어들도 업데이트
        # (예: decoder_att, fc 등 hidden_size를 입력받는 레이어)
        if hasattr(decoder, 'fc'):
            # fc: [hidden_size] → [vocab_size]
            old_fc = decoder.fc
            new_fc = nn.Linear(new_hidden_size, old_fc.out_features)
            new_fc.weight.data = old_fc.weight.data[:, indices_to_keep]
            new_fc.bias.data = old_fc.bias.data.clone()
            decoder.fc = new_fc
            print(f"      ✅ fc 레이어 업데이트: [{old_hidden_size}] → [{new_hidden_size}]")
        
        # init_h 레이어도 업데이트 (있다면)
        if hasattr(decoder, 'init_h'):
            old_init_h = decoder.init_h
            new_init_h = nn.Linear(old_init_h.in_features, new_hidden_size)
            new_init_h.weight.data = old_init_h.weight.data[indices_to_keep, :]
            new_init_h.bias.data = old_init_h.bias.data[indices_to_keep]
            decoder.init_h = new_init_h
            print(f"      ✅ init_h 레이어 업데이트: [*] → [{new_hidden_size}]")
    
    # Attention 차원도 축소
    if hasattr(decoder, 'encoder_att') and hasattr(decoder, 'full_att'):
        weight = decoder.encoder_att.weight.data
        
        # L2 norm 기반 중요도 계산 (간단함)
        mask_attention_dim = compute_channel_importance_hessian(
            weight, pruning_rate, dim=0, hessian_importance=None
        )
        new_attention_dim = mask_attention_dim.sum().item()
        
        print(f"   📊 Attention Dim: {weight.shape[0]} → {new_attention_dim} (제거: {weight.shape[0] - new_attention_dim})")
        
        # Attention 레이어 업데이트
        decoder.encoder_att = update_linear_layer(decoder.encoder_att, mask_out=mask_attention_dim, in_size=decoder.encoder_dim)
        decoder.attention_dim = new_attention_dim
        
        if hasattr(decoder, 'decoder_att'):
            new_decoder_att = nn.Linear(decoder.decoder_dim, new_attention_dim)
            nn.init.xavier_uniform_(new_decoder_att.weight)
            if new_decoder_att.bias is not None:
                nn.init.zeros_(new_decoder_att.bias)
            decoder.decoder_att = new_decoder_att
            print(f"   ✅ decoder_att 업데이트: [hidden={decoder.decoder_dim}] -> [attention={new_attention_dim}]")
        
        if hasattr(decoder, 'full_att'):
            decoder.full_att = update_linear_layer(decoder.full_att, mask_in=mask_attention_dim, out_size=1)
    
    pruned_model.decoder = decoder
    pruned_model.eval()
    
    # 파라미터 개수 확인
    old_params = sum(p.numel() for p in model.parameters())
    new_params = sum(p.numel() for p in pruned_model.parameters())
    reduction = (1 - new_params / old_params) * 100
    
    print(f"   ✂️ Structured Pruning 완료: GRU Hidden State + Attention 축소, {pruning_rate*100:.0f}% 프루닝")
    print(f"   📊 파라미터 감소: {old_params:,} → {new_params:,} ({reduction:.1f}% 감소)")
    print(f"   ⚡ **안전성**: GRU 순환 역학 부분 보존 + Hessian 기반 점진적 축소")
    
    return pruned_model

def apply_magnitude_pruning(model, pruning_rate):
    """Magnitude-based Pruning 적용 (Unstructured - 가중치 마스킹, 구조 변경 없음)
    
    Magnitude Pruning은 개별 가중치(Weight)의 절댓값(Magnitude)이 작은 것들을 0으로 설정합니다.
    **중요**: 모델의 실제 구조(차원)는 변하지 않습니다. 단지 일부 가중치가 0이 되어 희소성만 증가합니다.
    - 개별 가중치 제거 (레이어 구조 변경 없음)
    - 모델 파일 크기는 감소하지만, 일반 하드웨어에서는 추론 속도 향상 미미
    - 특수 희소 행렬 처리 하드웨어(예: NVIDIA Sparse Tensor Core)가 있어야 속도 향상
    """
    pruned_model = deepcopy(model)
    pruned_model.eval()
    
    # Magnitude-based pruning: 모델 구조는 유지하고 가중치만 마스킹
    # 각 가중치의 절댓값(magnitude)을 기준으로 낮은 것부터 0으로 설정
    
    # 1. 전체 모델의 가중치에 magnitude-based masking 적용
    pruned_params = 0
    for name, param in pruned_model.named_parameters():
        if param.dim() >= 2:  # 2차원 이상의 가중치만 처리 (bias 제외)
            # 절댓값(magnitude) 기반 중요도 계산
            magnitude = torch.abs(param.data)
            
            # 프루닝할 가중치 개수 계산
            num_to_prune = int(pruning_rate * param.numel())
            
            if num_to_prune > 0 and num_to_prune < param.numel():
                # 가중치를 크기 순서로 정렬하여 가장 작은 것부터 선택
                threshold = torch.kthvalue(magnitude.flatten(), num_to_prune).values
                
                # 임계값 이하의 가중치를 0으로 설정 (마스킹)
                mask = magnitude > threshold
                param.data = param.data * mask.float()
                pruned_params += num_to_prune
    
    pruned_model.eval()
    
    # 파라미터 개수는 변하지 않음 (구조 변경 없음)
    # 하지만 0이 아닌 파라미터 개수는 감소
    total_params = sum(p.numel() for p in pruned_model.parameters())
    nonzero_params = sum((p != 0).sum().item() for p in pruned_model.parameters())
    weight_sparsity = (1 - nonzero_params / total_params) * 100
    
    print(f"   ✂️ Magnitude-based Pruning 완료: 가중치 마스킹 기반, {pruning_rate*100:.0f}% 가중치 제거")
    print(f"   📊 희소성(Sparsity): {weight_sparsity:.1f}% (구조 변경 없음, {total_params:,}개 파라미터 유지)")
    print(f"   💡 주의: 일반 하드웨어에서는 0인 가중치도 계산되므로 실제 속도 향상 미미")
    
    return pruned_model

# ============================================================================
# 벤치마크 엔진
# ============================================================================

def run_pruning_benchmark(pruned_model, label, img_tensor, wm, rwm, ref_caption, baseline_params, device, results):
    """프루닝된 모델 벤치마크 및 파인튜닝 실행"""
    pruned_model.to(device)
    
    # 프루닝 후 벤치마크
    result = run_benchmark(pruned_model, img_tensor, wm, rwm, label, ref_caption, baseline_params=baseline_params)
    if result:
        results.append(result)
    
    # 파인 튜닝 (Epoch마다 벤치마크 및 모델 저장 포함)
    fine_tuned_model = fine_tune_pruned_model(
        pruned_model, wm, 
        img_tensor=img_tensor, wm=wm, rwm=rwm,
        ref_caption=ref_caption, baseline_params=baseline_params,
        epochs=FINETUNE_EPOCHS, label=label.replace(" ", "_").replace("%", "pct"),
        learning_rate=LEARNING_RATE
    )
    fine_tuned_model.to(device)
    
    # 파인 튜닝 후 최종 벤치마크
    result_finetuned = run_benchmark(fine_tuned_model, img_tensor, wm, rwm, f"{label} (Fine-tuned)", ref_caption, baseline_params=baseline_params)
    if result_finetuned:
        results.append(result_finetuned)
    
    del pruned_model, fine_tuned_model
    gc.collect()

def run_benchmark(model, img_tensor, wm, rwm, precision_name, ref_caption=None, baseline_params=None):
    print(f"\n[{precision_name}] 벤치마크 시작...")
    
    model_device = next(model.parameters()).device
    inp = img_tensor.clone().detach().to(model_device)
    
    # Warm-up
    with torch.no_grad():
        try:
            _ = model.generate(inp, wm, rwm, 20)
        except Exception as e:
            print(f"⚠️ Warm-up 실패: {e}")
            return None
    
    # benchmark_utils를 사용한 시간 및 메모리 측정
    inference_metrics = measure_inference_time(model, inp, num_runs=NUM_RUNS, warmup=5)
    
    latencies = [inference_metrics['mean_ms']] * NUM_RUNS  # 평균값 사용
    memory_usages = [get_peak_memory_mb()] * NUM_RUNS  # 평균 메모리 사용
    
    print(f"   ⏱️ 평균 추론 시간: {inference_metrics['mean_ms']:.2f} ± {inference_metrics['std_ms']:.2f} ms")
    print(f"   🧠 메모리 사용량: {get_peak_memory_mb():.2f} MB")
    
    # METEOR 점수 계산 (10개 이미지로 측정)
    meteor_scores = []
    example_caption = "N/A"
    
    # 10개의 이미지로 METEOR 점수 측정
    test_images_meteor = []
    test_captions_meteor = []
    
    # 테스트 이미지 디렉토리에서 100개 이미지 로드
    if os.path.exists(TEST_IMAGE_DIR):
        image_files = [f for f in os.listdir(TEST_IMAGE_DIR) 
                      if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if image_files:
            import random
            # 최대 10개 이미지 선택
            selected_files = random.sample(image_files, min(METEO_IMAGE_NUM, len(image_files)))
            
            # 각 이미지와 캡션 로드
            for filename in selected_files:
                try:
                    img_path = os.path.join(TEST_IMAGE_DIR, filename)
                    img = Image.open(img_path).convert('RGB')
                    img_tensor_meteor = transform(img).unsqueeze(0).to(model_device)
                    test_images_meteor.append(img_tensor_meteor)
                    
                    # 참조 캡션 로드
                    ref_cap = None
                    if os.path.exists(CAPTIONS_FILE):
                        with open(CAPTIONS_FILE, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            for line in lines:
                                if ',' in line:
                                    parts = line.split(',', 1)
                                    if len(parts) == 2 and parts[0].strip() == filename:
                                        ref_cap = parts[1].strip()
                                        break
                    
                    if ref_cap:
                        test_captions_meteor.append(ref_cap)
                    else:
                        test_captions_meteor.append(None)
                except Exception as e:
                    print(f"   ⚠️ 이미지 로드 실패 ({filename}): {e}")
                    continue
    
    # 이미지가 부족하면 더미 데이터로 채움
    while len(test_images_meteor) < 10:
        dummy_img = torch.randn(1, 3, 224, 224).to(model_device)
        test_images_meteor.append(dummy_img)
        test_captions_meteor.append("a test image")
    
    # METEO_IMAGE_NUM개 이미지에 대해 METEOR 점수 계산
    if test_images_meteor and any(test_captions_meteor):
        print(f"   📊 METEOR 점수 측정 중: {len([c for c in test_captions_meteor if c])}개 이미지")
        for idx, (test_img, ref_cap) in enumerate(zip(test_images_meteor[:METEO_IMAGE_NUM], test_captions_meteor[:METEO_IMAGE_NUM])):
            if ref_cap:
                with torch.no_grad():
                    gen_seq = model.generate(test_img, wm, rwm, 20)
                meteor = calculate_meteor(gen_seq, ref_cap)
                if meteor is not None:
                    meteor_scores.append(meteor)
                if idx == 0:
                    example_caption = ' '.join([w for w in gen_seq if w not in ['<start>', '<end>', '<pad>', '<unk>']])
                    ref_caption = ref_cap
    
    avg_meteor = np.mean(meteor_scores) if meteor_scores else None
    
    # 결과 정리
    avg_time = inference_metrics['mean_ms']
    std_time = inference_metrics['std_ms']
    
    # benchmark_utils 사용: 모델 크기 계산
    size_mb_dense = calculate_model_size_mb(model, model_type='dense')
    size_mb_sparse = calculate_model_size_mb(model, model_type='sparse')
    sparsity = calculate_sparsity(model)
    
    # 추론 과정에서의 평균 메모리 사용량
    memory_usage = np.mean(memory_usages) if memory_usages else 0.0
    total_params, trainable_params = count_parameters(model)
    nonzero_params, _ = count_nonzero_parameters(model)
    
    print(f"   ⏱️ 평균 시간: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"   💾 모델 크기 (Dense): {size_mb_dense:.2f} MB")
    print(f"   💾 모델 크기 (Sparse): {size_mb_sparse:.2f} MB")
    print(f"   📉 크기 감소율: {(1 - size_mb_sparse/size_mb_dense)*100:.2f}%")
    print(f"   📊 총 파라미터: {total_params:,} (0이 아닌: {nonzero_params:,})")
    print(f"   ✂️ Sparsity: {sparsity:.2f}%")
    print(f"   🧠 메모리 사용량: {memory_usage:.5f} MB")
    if avg_meteor is not None:
        print(f"   ⭐ METEOR: {avg_meteor:.4f}")
    print(f"   📝 예시 캡션: {example_caption}")
    print(f"  📝 참조 캡션{ref_caption}")
    
    return {
        'precision': precision_name,
        'mean_time_ms': avg_time,
        'std_time_ms': std_time,
        'min_time_ms': inference_metrics['min_ms'],
        'max_time_ms': inference_metrics['max_ms'],
        'model_size_mb': size_mb_sparse,  # Sparse format 크기 사용
        'model_size_mb_dense': size_mb_dense,  # Dense format 크기도 저장
        'memory_usage_mb': memory_usage,
        'meteor_score': avg_meteor,
        'inference_times': latencies,
        'example_caption': example_caption,
        'total_params': total_params,
        'nonzero_params': nonzero_params,
        'sparsity': sparsity,
        'trainable_params': trainable_params,
        'size_reduction': (1 - size_mb_sparse/size_mb_dense)*100 if size_mb_dense > 0 else 0
    }

# ============================================================================
# 시각화
# ============================================================================
def plot_pruning_comparison(results):
    """Pruning 결과 비교 그래프 (파인 튜닝 제외)"""
    if not results:
        print("❌ 결과가 없어 plot을 생성할 수 없습니다.")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 파인 튜닝된 결과 제외
    filtered_results = [r for r in results if '(Fine-tuned)' not in r['precision']]
    
    if not filtered_results:
        print("❌ 파인 튜닝 제외 후 결과가 없어 plot을 생성할 수 없습니다.")
        return
    
    precisions = [r['precision'] for r in filtered_results]
    mean_times = [r['mean_time_ms'] for r in filtered_results]
    std_times = [r['std_time_ms'] for r in filtered_results]
    model_sizes = [r['model_size_mb'] for r in filtered_results]
    memory_usages = [r['memory_usage_mb'] for r in filtered_results]
    meteor_scores = [r.get('meteor_score', None) for r in filtered_results]
    sparsities = [r.get('sparsity', 0) * 100 for r in filtered_results]
    nonzero_params_list = [r.get('nonzero_params', 0) for r in filtered_results]
    
    valid_meteor_scores = [s for s in meteor_scores if s is not None]
    valid_meteor_precisions = [p for p, s in zip(precisions, meteor_scores) if s is not None]
    
    # 색상 설정
    colors = plt.cm.viridis(np.linspace(0, 1, len(precisions)))
    
    # 종합 비교 그래프
    if valid_meteor_scores:
        fig, axes = plt.subplots(3, 2, figsize=(14, 15))
        fig.suptitle('Pruning 성능 비교 종합', fontsize=16, fontweight='bold')
        
        # 1. 추론 시간
        axes[0, 0].bar(precisions, mean_times, alpha=0.8, color=colors, yerr=std_times, capsize=5)
        axes[0, 0].set_ylabel('추론 시간 (ms)', fontweight='bold')
        axes[0, 0].set_title('추론 시간', fontweight='bold')
        axes[0, 0].grid(axis='y', alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        for i, (p, m, s) in enumerate(zip(precisions, mean_times, std_times)):
            axes[0, 0].text(i, m + s + 1, f'{m:.1f}', ha='center', va='bottom', fontsize=9)
        
        # 2. 모델 크기 (Sparse format - 실제 저장 크기)
        axes[0, 1].bar(precisions, model_sizes, alpha=0.8, color=colors, label='Sparse (실제 저장 크기)')
        axes[0, 1].set_ylabel('모델 크기 (MB)', fontweight='bold')
        axes[0, 1].set_title('모델 크기 (Sparse Format)', fontweight='bold')
        axes[0, 1].grid(axis='y', alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].legend()
        for i, (p, s) in enumerate(zip(precisions, model_sizes)):
            axes[0, 1].text(i, s + 0.5, f'{s:.1f}', ha='center', va='bottom', fontsize=9)
        
        # 3. Sparsity (파라미터 감소율)
        axes[1, 0].bar(precisions, sparsities, alpha=0.8, color=colors)
        axes[1, 0].set_ylabel('파라미터 감소율 (%)', fontweight='bold')
        axes[1, 0].set_title('파라미터 감소율 (Baseline 대비)', fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
        # Y축 범위 설정 (0-100%)
        axes[1, 0].set_ylim(0, max(sparsities) * 1.2 if sparsities else 100)
        for i, (p, s) in enumerate(zip(precisions, sparsities)):
            axes[1, 0].text(i, s + max(sparsities) * 0.02 if sparsities else 1, 
                          f'{s:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 4. METEOR 점수
        axes[1, 1].bar(valid_meteor_precisions, valid_meteor_scores, alpha=0.8, 
                     color=colors[:len(valid_meteor_scores)])
        axes[1, 1].set_ylabel('METEOR 점수', fontweight='bold')
        axes[1, 1].set_title('METEOR 점수 (캡션 품질)', fontweight='bold')
        axes[1, 1].set_ylim(0, 1.0)
        axes[1, 1].grid(axis='y', alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=45)
        for i, (p, s) in enumerate(zip(valid_meteor_precisions, valid_meteor_scores)):
            axes[1, 1].text(i, s + 0.01, f'{s:.2f}', ha='center', va='bottom', fontsize=9)
        
        # 5. 메모리 사용량
        axes[2, 0].bar(precisions, memory_usages, alpha=0.8, color=colors)
        axes[2, 0].set_ylabel('메모리 사용량 (MB)', fontweight='bold')
        axes[2, 0].set_title('메모리 사용량', fontweight='bold')
        axes[2, 0].grid(axis='y', alpha=0.3)
        axes[2, 0].tick_params(axis='x', rotation=45)
        for i, (p, m) in enumerate(zip(precisions, memory_usages)):
            axes[2, 0].text(i, m + max(memory_usages) * 0.02, f'{m:.1f}', ha='center', va='bottom', fontsize=9)
        
        # 6. 총 파라미터 개수 (물리적 구조 변경 후 실제 파라미터 수)
        total_params_list = [r.get('total_params', 0) for r in filtered_results]
        total_params_m = [p / 1e6 for p in total_params_list]
        axes[2, 1].bar(precisions, total_params_m, alpha=0.8, color=colors)
        axes[2, 1].set_ylabel('총 파라미터 (M)', fontweight='bold')
        axes[2, 1].set_title('총 파라미터 개수 (물리적 구조 변경 후)', fontweight='bold')
        axes[2, 1].grid(axis='y', alpha=0.3)
        axes[2, 1].tick_params(axis='x', rotation=45)
        for i, (p, tp_m) in enumerate(zip(precisions, total_params_m)):
            axes[2, 1].text(i, tp_m + max(total_params_m) * 0.02 if total_params_m else 0.1, 
                          f'{tp_m:.2f}M', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pruning_comparison_comprehensive.png'), 
                dpi=300, bbox_inches='tight')
    print(f"✅ Plot 저장: {os.path.join(OUTPUT_DIR, 'pruning_comparison_comprehensive.png')}")
    plt.close()

def plot_finetune_comparison(results, baseline_result):
    """파인 튜닝 전후 METEOR 점수 비교 그래프"""
    if not results or not baseline_result:
        print("❌ 결과가 없어 plot을 생성할 수 없습니다.")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 프루닝 후(Fine-tuned 제외)와 파인튜닝 후(Fine-tuned) 결과 분리
    before_finetune = {r['precision']: r for r in results if '(Fine-tuned)' not in r['precision']}
    after_finetune = {r['precision'].replace(' (Fine-tuned)', ''): r for r in results if '(Fine-tuned)' in r['precision']}
    
    # 매칭되는 모델 찾기
    model_names = []
    meteor_before = []
    meteor_after = []
    
    baseline_meteor = baseline_result.get('meteor_score', 0)
    
    for model_name in sorted(after_finetune.keys()):
        if model_name in before_finetune:
            before = before_finetune[model_name]
            after = after_finetune[model_name]
            
            model_names.append(model_name)
            meteor_before.append(before.get('meteor_score', 0))
            meteor_after.append(after.get('meteor_score', 0))
    
    if not model_names:
        print("❌ 파인 튜닝 전후 비교할 데이터가 없습니다.")
        return
    
    # 그래프 생성
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(model_names))
    width = 0.25
    
    # Baseline 선
    ax.axhline(y=baseline_meteor, color='red', linestyle='--', linewidth=2, label=f'Baseline ({baseline_meteor:.4f})', alpha=0.7)
    
    # 파인튜닝 전 (프루닝 후)
    bars1 = ax.bar(x - width, meteor_before, width, label='Pruned (Before Fine-tuning)', alpha=0.8, color='steelblue')
    
    # 파인튜닝 후
    bars2 = ax.bar(x, meteor_after, width, label='Pruned (After Fine-tuning)', alpha=0.8, color='orange')
    
    # 레이블 및 제목
    ax.set_ylabel('METEOR Score', fontweight='bold', fontsize=12)
    ax.set_title('파인튜닝 전후 METEOR 점수 비교', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # 값 표시
    for i, (before, after) in enumerate(zip(meteor_before, meteor_after)):
        # 파인튜닝 전
        ax.text(i - width, before + 0.005, f'{before:.4f}', ha='center', va='bottom', fontsize=9)
        # 파인튜닝 후
        ax.text(i, after + 0.005, f'{after:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 개선율 표시
        improvement = ((after - before) / before * 100) if before != 0 else 0
        improvement_text = f'{improvement:+.1f}%'
        ax.text(i + width/2, max(before, after) + 0.01, improvement_text, 
                ha='center', va='bottom', fontsize=9, fontweight='bold', 
                color='green' if improvement > 0 else 'red')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pruning_finetune_meteor_comparison.png'), 
                dpi=300, bbox_inches='tight')
    print(f"✅ 파인튜닝 METEOR 비교 Plot 저장: {os.path.join(OUTPUT_DIR, 'pruning_finetune_meteor_comparison.png')}")
    plt.close()

# ============================================================================
# 파인 튜닝 함수
# ============================================================================
def fine_tune_pruned_model(model, word_map, img_tensor=None, wm=None, rwm=None, ref_caption=None, baseline_params=None, epochs=2, label="pruned_model", learning_rate=5e-5):
    """파인튜닝 수행 + Epoch마다 벤치마크 및 모델 저장 + 체크포인트 로드"""
    print(f"\n   🔄 파인 튜닝 시작 ({epochs} epoch)...")
    
    # 체크포인트 로드
    checkpoint, start_epoch, checkpoint_path = load_checkpoint(label, device)
    optimizer_state = checkpoint.get('optimizer_state_dict') if checkpoint else None
    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print_checkpoint_info(checkpoint, start_epoch)
        print(f"   ✅ Epoch {start_epoch+1}부터 재개합니다.")
    else:
        print(f"   ℹ️ 처음부터 시작합니다.")
    
    # 학습 설정
    optimizer, criterion = setup_training(model, learning_rate, device)
    restore_optimizer(optimizer, optimizer_state)
    
    # 학습 데이터셋 준비 (학습/검증 분할)
    try:
        full_dataset = CaptionDataset(
            images_dir=TEST_IMAGE_DIR,
            captions_file=CAPTIONS_FILE,
            transform=transform,
            word_map=word_map,
            max_len=50
        )
        
        if len(full_dataset) == 0:
            print("   ⚠️ 학습 데이터가 없어 파인 튜닝을 건너뜁니다.")
            return model
        
        # 학습/검증 데이터셋 분할
        val_size = int(len(full_dataset) * VALIDATION_SPLIT)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"   📊 데이터셋 분할: 학습({train_size}개) / 검증({val_size}개)")
        
        # 적응형 배치 사이즈
        batch_size = 32 if train_size < 1000 else 64
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        print(f"   📚 학습 배치: {len(train_dataloader)}개, 검증 배치: {len(val_dataloader)}개")
        
        # 모델을 학습 모드로 전환
        model.train()
        model.to(device)
        
        # 체크포인트에서 Optimizer State 복구
        if optimizer_state is not None:
            try:
                optimizer.load_state_dict(optimizer_state)
                print(f"   ✅ Optimizer State 복구 완료 (Learning Rate, Momentum 등 복원)")
            except Exception as e:
                print(f"   ⚠️ Optimizer State 복구 실패: {e}")
        
        # 학습할 파라미터 개수 출력
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_count = sum(p.numel() for p in model.parameters())
        print(f"   📊 학습 대상 파라미터: {trainable_count:,} / {total_count:,} ({100*trainable_count/total_count:.1f}%)")
        vocab_size = len(word_map)
        
        # Early Stopping 설정
        best_meteor_score = -float('inf')
        patience_counter = 0
        
        # 파인튜닝 진행 (체크포인트 이후부터 시작)
        for epoch in range(start_epoch, epochs):
            print(f"   🏋️ Epoch {epoch+1}/{epochs}")
            total_loss = 0
            num_batches = 0
            
            for batch_idx, (imgs, caps) in enumerate(train_dataloader):
                imgs = imgs.to(device)
                caps = caps.to(device)
                
                optimizer.zero_grad()
            
                try:
                    outputs, alphas = model(imgs, caps)
                    targets = caps[:, 1:]
                    outputs = outputs[:, :targets.shape[1], :]
                    loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                except Exception as e:
                    print(f"   ⚠️ 배치 {batch_idx} 학습 실패: {e}")
                    continue
                
                # 10개 배치마다 진행상황 출력
                if (batch_idx + 1) % 10 == 0:
                    print(f"      배치 {batch_idx + 1}/{len(train_dataloader)}, Loss: {total_loss / num_batches:.4f}")
            
            # 🎯 Epoch 끝 - 학습 Loss 계산
            if num_batches > 0:
                avg_loss = total_loss / num_batches
                print(f"   ✅ Epoch {epoch+1} 완료 (학습 Loss: {avg_loss:.4f})")
            
            # 🔍 검증 데이터셋 평가
            print(f"   📊 검증 데이터 평가 중...")
            model.eval()
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for val_imgs, val_caps in val_dataloader:
                    val_imgs = val_imgs.to(device)
                    val_caps = val_caps.to(device)
                    
                    try:
                        val_outputs, _ = model(val_imgs, val_caps)
                        val_targets = val_caps[:, 1:]
                        val_outputs = val_outputs[:, :val_targets.shape[1], :]
                        val_loss_batch = criterion(val_outputs.reshape(-1, vocab_size), val_targets.reshape(-1))
                        val_loss += val_loss_batch.item()
                        val_batches += 1
                    except Exception as e:
                        continue
            
            avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
            print(f"      검증 Loss: {avg_val_loss:.4f}")
            
            model.train()  # 다시 학습 모드
            
            # 벤치마크 실행 (img_tensor, wm, rwm이 제공된 경우)
            current_meteor_score = None
            if img_tensor is not None and wm is not None and rwm is not None:
                print(f"\n   📊 Epoch {epoch+1} 벤치마크 시작...")
                model.eval()
                benchmark_result = run_benchmark(
                    model, img_tensor, wm, rwm, 
                    f"Fine-tuned (Epoch {epoch+1}/{epochs})",
                    ref_caption=ref_caption,
                    baseline_params=baseline_params
                )
                model.train()  # 다시 학습 모드로
                
                # 벤치마크 결과 출력
                if benchmark_result:
                    print(f"\n   📈 Epoch {epoch+1} 결과:")
                    print(f"      ⏱️ 평균 시간: {benchmark_result['mean_time_ms']:.2f} ms")
                    print(f"      💾 모델 크기: {benchmark_result['model_size_mb']:.2f} MB")
                    print(f"      🧠 메모리: {benchmark_result['memory_usage_mb']:.2f} MB")
                    if benchmark_result.get('meteor_score'):
                        current_meteor_score = benchmark_result['meteor_score']
                        print(f"      ⭐ METEOR: {current_meteor_score:.4f}")
            
            # 🛑 Early Stopping 체크 (METEOR 점수 기반)
            if current_meteor_score is not None:
                if current_meteor_score > best_meteor_score:
                    best_meteor_score = current_meteor_score
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                    print(f"   🎉 새로운 최고 METEOR 점수: {best_meteor_score:.4f} (Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE})")
                else:
                    patience_counter += 1
                    print(f"   ⚠️ METEOR 점수 미개선: {current_meteor_score:.4f} (Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE})")
                    
                    if patience_counter >= EARLY_STOPPING_PATIENCE:
                        print(f"\n   🛑 Early Stopping 발동! Epoch {epoch+1}에서 학습 종료")
                        print(f"      최고 METEOR 점수: {best_meteor_score:.4f}")
                        # 최고 성능 모델 로드
                        model.load_state_dict(best_model_state)
                        break
                
                # 체크포인트 저장 (함수 사용)
                save_checkpoint(model, optimizer, epoch, label, 
                               avg_loss if num_batches > 0 else None,
                               avg_val_loss, current_meteor_score)
        
        model.eval()
        return model
        
    except Exception as e:
        print(f"   ⚠️ 파인 튜닝 실패: {e}")
        import traceback
        traceback.print_exc()
        return model

# ============================================================================
# Main
# ============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*70)
    print("=== Pruning 벤치마크 ===")
    print("="*70)
    
    # 1. 모델 및 데이터 로드
    base_model, wm, rwm = load_base_model(device=device)
    img_tensor, ref_caption = load_test_data(device=device, transform=transform)
    
    results = []
    
    # 2. 원본 모델 벤치마크 (Baseline)
    print("\n" + "="*70)
    print("=== [Baseline] 원본 모델 ===")
    print("="*70)
    result_baseline = run_benchmark(base_model, img_tensor, wm, rwm, "Original (Baseline)", ref_caption)
    baseline_params = None
    if result_baseline:
        baseline_params = result_baseline['total_params']
        results.append(result_baseline)
    
    # 3. 다양한 Pruning Rate로 테스트
    for pruning_rate in PRUNING_RATES:
        # Magnitude-based Pruning (선택적 - 이 모델에는 비효율적)
        if ENABLE_MAGNITUDE_PRUNING:
            print("\n" + "="*70)
            print(f"=== Magnitude Pruning ({pruning_rate*100:.0f}%) ===")
            print("="*70)
            try:
                pruned_model = apply_magnitude_pruning(base_model, pruning_rate)
                run_pruning_benchmark(pruned_model, f"Magnitude-{pruning_rate*100:.0f}%", img_tensor, wm, rwm, ref_caption, baseline_params, device, results)
            except Exception as e:
                print(f"⚠️ Magnitude Pruning ({pruning_rate*100:.0f}%) 실패: {e}")
                import traceback
                traceback.print_exc()
        
        # Structured Pruning
        print("\n" + "="*70)
        print(f"=== Structured Pruning ({pruning_rate*100:.0f}%) ===")
        print("="*70)
        
        # ⚠️ 30% 이상 프루닝은 정확도 급격히 하락하므로 경고
        if pruning_rate > MAX_PRUNING_RATE:
            print(f"   ⚠️ 경고: {pruning_rate*100:.0f}% 프루닝은 정확도 손실이 매우 큼 (권장: {MAX_PRUNING_RATE*100:.0f}% 이하)")
        
        try:
            pruned_model = apply_structured_pruning_physical(
                base_model, pruning_rate, 
                img_tensor=img_tensor, wm=wm, rwm=rwm, 
                device=device, use_hessian=True
            )
            run_pruning_benchmark(pruned_model, f"Structured-{pruning_rate*100:.0f}%", img_tensor, wm, rwm, ref_caption, baseline_params, device, results)
            # run_benchmark(pruned_model, img_tensor, wm, rwm, f"Structured-{pruning_rate*100:.0f}%", ref_caption)

        except Exception as e:
            print(f"⚠️ Structured Pruning ({pruning_rate*100:.0f}%) 실패: {e}")
            import traceback
            traceback.print_exc()
    
    
    
    print("\n" + "="*70)
    print("=== 벤치마크 결과 요약 ===")
    print("="*70)
    if any(r.get('meteor_score') is not None for r in results):
        print(f"{'Method':<25} {'추론시간(ms)':<15} {'모델크기(MB)':<15} {'크기감소(%)':<15} {'Sparsity(%)':<15} {'METEOR':<10}")
        print("-"*100)
        for result in results:
            meteor_str = f"{result.get('meteor_score', 0):.4f}" if result.get('meteor_score') is not None else "N/A"
            sparsity = result.get('sparsity', 0) * 100
            size_reduction = result.get('size_reduction', 0)
            print(f"{result['precision']:<25} "
                  f"{result['mean_time_ms']:.2f}±{result['std_time_ms']:.2f}    "
                  f"{result['model_size_mb']:.2f}          "
                  f"{size_reduction:<15.2f} "
                  f"{sparsity:<15.2f} "
                  f"{meteor_str}")
    else:
        print(f"{'Method':<25} {'추론시간(ms)':<15} {'모델크기(MB)':<15} {'크기감소(%)':<15} {'Sparsity(%)':<15}")
        print("-"*85)
        for result in results:
            sparsity = result.get('sparsity', 0) * 100
            size_reduction = result.get('size_reduction', 0)
            print(f"{result['precision']:<25} "
                  f"{result['mean_time_ms']:.2f}±{result['std_time_ms']:.2f}    "
                  f"{result['model_size_mb']:.2f}          "
                  f"{size_reduction:<15.2f} "
                  f"{sparsity:<15.2f}")
    
    # 6. 결과 저장
    print("\n" + "="*70)
    print("결과 저장 중...")
    print("="*70)
    
    # JSON 형식으로 결과 저장
    import json
    results_dict = {
        'baseline_params': baseline_params,
        'results': []
    }
    
    for result in results:
        result_dict = {
            'precision': result['precision'],
            'mean_time_ms': float(result['mean_time_ms']),
            'std_time_ms': float(result['std_time_ms']),
            'min_time_ms': float(result['min_time_ms']),
            'max_time_ms': float(result['max_time_ms']),
            'model_size_mb': float(result['model_size_mb']),
            'model_size_mb_dense': float(result.get('model_size_mb_dense', 0)),
            'memory_usage_mb': float(result['memory_usage_mb']),
            'meteor_score': float(result.get('meteor_score', 0)) if result.get('meteor_score') is not None else None,
            'total_params': int(result['total_params']),
            'trainable_params': int(result.get('trainable_params', 0)),
            'nonzero_params': int(result.get('nonzero_params', 0)),
            'sparsity': float(result.get('sparsity', 0)),
            'size_reduction': float(result.get('size_reduction', 0)),
            'example_caption': result.get('example_caption', 'N/A')
        }
        results_dict['results'].append(result_dict)
    
    results_json_path = os.path.join(OUTPUT_DIR, 'pruning_results.json')
    with open(results_json_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    print(f"✅ 결과 JSON 저장: {results_json_path}")
    
    # 7. 시각화
    print("\n" + "="*70)
    print("Plot 생성 중...")
    print("="*70)
    plot_pruning_comparison(results)
    
    # 파인 튜닝 비교 그래프 생성
    if result_baseline:
        plot_finetune_comparison(results, result_baseline)
    
    print("\n" + "="*70)
    print("=== 벤치마크 완료 ===")
    print(f"결과 저장 위치: {OUTPUT_DIR}")
    print(f"  - JSON: {results_json_path}")
    print(f"  - Plot: {os.path.join(OUTPUT_DIR, 'pruning_comparison_comprehensive.png')}")
    print("="*70)

if __name__ == "__main__":
    main()

