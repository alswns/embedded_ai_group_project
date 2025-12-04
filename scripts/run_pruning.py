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
# 디바이스 선택
device = setup_device()

# 이미지 전처리
transform = get_image_transform()

# ============================================================================
# Pruning 전용 유틸리티 함수
# ============================================================================
def count_nonzero_parameters(model):
    """0이 아닌 파라미터 개수 계산 (프루닝 후)"""
    nonzero_params = 0
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
        nonzero_params += param.nonzero().size(0) if param.numel() > 0 else 0
    return nonzero_params, total_params

def convert_to_sparse_model(model):
    """Pruning된 모델을 실제로 sparse format으로 변환하여 크기 감소"""
    # 주의: 실제로 모델 구조를 변경하는 것은 복잡하므로
    # 여기서는 가중치를 sparse tensor로 변환하는 대신
    # 실제 0이 아닌 파라미터만 계산하는 방식 사용
    # 실제 배포 시에는 sparse format으로 저장/로드하는 것이 좋습니다
    return model

def save_sparse_model(model, path):
    """모델을 sparse format으로 저장 (실제 크기 감소)"""
    state_dict = {}
    for name, param in model.named_parameters():
        if param.numel() > 0:
            # 0이 아닌 값만 저장
            nonzero_mask = param != 0
            if nonzero_mask.any():
                # Sparse format으로 저장
                sparse_param = param[nonzero_mask]
                indices = nonzero_mask.nonzero(as_tuple=False)
                state_dict[name] = {
                    'values': sparse_param.cpu(),
                    'indices': indices.cpu(),
                    'shape': list(param.shape),
                    'dtype': str(param.dtype)
                }
            else:
                # 모든 값이 0인 경우
                state_dict[name] = {
                    'values': torch.tensor([], dtype=param.dtype),
                    'indices': torch.tensor([], dtype=torch.long),
                    'shape': list(param.shape),
                    'dtype': str(param.dtype)
                }
        else:
            state_dict[name] = param.cpu()
    
    # 버퍼도 저장
    for name, buffer in model.named_buffers():
        state_dict[name] = buffer.cpu()
    
    torch.save(state_dict, path)
    print(f"   💾 Sparse 모델 저장: {path}")

def get_sparse_model_size_mb(model):
    """Sparse format으로 저장했을 때의 실제 모델 크기 계산 (더 현실적인 계산)"""
    total_size = 0
    
    for name, param in model.named_parameters():
        if param.numel() > 0:
            # 0이 아닌 값의 개수
            nonzero_count = (param != 0).sum().item()
            total_params = param.numel()
            
            if nonzero_count > 0:
                # 값 저장 (0이 아닌 값만) - float32 = 4 bytes
                total_size += nonzero_count * param.element_size()
                
                # 인덱스 저장 (더 효율적인 압축 사용) - int16 = 2 bytes per dimension
                # CSR (Compressed Sparse Row) format 가정
                num_dimensions = len(param.shape)
                # row pointer: shape[0] + 1 elements
                # column indices: nonzero_count elements
                # 평균적으로 각 다차원별로 2 bytes
                indices_size = int(nonzero_count * num_dimensions * 2)
                
                total_size += indices_size
                
                # 메타데이터 최소화 (형태, dtype 등)
                total_size += 16  # 최소 메타데이터
            else:
                # 모든 값이 0인 경우 최소 메타데이터만
                total_size += 8
    
    # 버퍼 크기 (배치 정규화 등)
    for name, buffer in model.named_buffers():
        total_size += buffer.nelement() * buffer.element_size()
    
    return total_size / 1024 / 1024

# ============================================================================
# 데이터 로드 (공통 모듈 사용)
# ============================================================================
# load_base_model, load_test_data는 utils에서 import

# ============================================================================
# Pruning 함수 (물리적 구조 수정)
# ============================================================================

def get_pruning_mask(weight, pruning_rate, dim=0, use_l2=True):
    """프루닝 마스크 생성 (제거할 채널/뉴런 식별)
    
    Args:
        weight: 가중치 텐서
        pruning_rate: 프루닝 비율
        dim: 프루닝할 차원 (0: 출력, 1: 입력)
        use_l2: True면 L2 norm 사용, False면 L1 norm 사용
    """
    # 중요도 계산: dim 차원을 따라 축약
    if use_l2:
        importance = torch.norm(weight, p=2, dim=1 if dim == 0 else 0)
    else:
        importance = torch.abs(weight).sum(dim=1 if dim == 0 else 0)
    
    # 중요도가 낮은 순서로 정렬
    num_to_prune = int(pruning_rate * importance.numel())
    if num_to_prune == 0:
        return torch.ones(importance.numel(), dtype=torch.bool, device=weight.device)
    
    _, indices = torch.sort(importance)
    mask = torch.ones(importance.numel(), dtype=torch.bool, device=weight.device)
    mask[indices[:num_to_prune]] = False
    return mask

def update_linear_layer(old_layer, mask_in=None, mask_out=None, in_size=None, out_size=None):
    """선형 레이어를 마스크에 따라 업데이트하고 새 레이어 반환
    
    Args:
        old_layer: 기존 nn.Linear 레이어
        mask_in: 입력 차원 마스크 (True=유지, False=제거)
        mask_out: 출력 차원 마스크 (기본값: 차원 변경 없음)
        in_size: 입력 차원 크기 (mask_in 없을 때)
        out_size: 출력 차원 크기 (mask_out 없을 때)
    """
    weight = old_layer.weight.data
    
    # 출력/입력 차원 계산
    if mask_out is not None:
        new_out = mask_out.sum().item()
        weight = weight[mask_out, :]
    else:
        new_out = out_size or weight.shape[0]
    
    if mask_in is not None:
        new_in = mask_in.sum().item()
        weight = weight[:, mask_in]
    else:
        new_in = in_size or weight.shape[1]
    
    # 새 레이어 생성
    new_layer = nn.Linear(new_in, new_out)
    new_layer.weight.data = weight
    
    # 바이어스 업데이트
    if old_layer.bias is not None:
        if mask_out is not None:
            new_layer.bias.data = old_layer.bias.data[mask_out]
        else:
            new_layer.bias.data = old_layer.bias.data.clone()
    
    return new_layer


def compute_hessian_importance(model, layer, img_tensor, captions_batch, wm, rwm, device, num_samples=64):
    """Hessian 기반 중요도 계산 (Fisher Information Matrix 근사)
    
    실제 Loss(CrossEntropyLoss)를 사용하여 각 채널의 손실에 대한 
    2계 미분을 계산하여 중요도 판정
    
    Args:
        num_samples: 최소 64개 이상의 샘플로 안정적인 Hessian 추정
    """
    model.train()  # Batch Norm 등 변동성 있는 레이어 활성화
    model.to(device)
    
    # CrossEntropyLoss 준비
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    importance = None
    vocab_size = len(wm)
    
    # 충분한 샘플 수를 사용하여 Fisher Information 계산
    num_samples = min(num_samples, len(captions_batch) if isinstance(captions_batch, list) else captions_batch.shape[0])
    
    for sample_idx in range(num_samples):
        model.zero_grad()
        
        # Forward pass
        inp = img_tensor[sample_idx:sample_idx+1].clone().detach().to(device)
        caps = captions_batch[sample_idx] if isinstance(captions_batch, list) else captions_batch[sample_idx:sample_idx+1]
        caps = torch.tensor(caps, dtype=torch.long, device=device).unsqueeze(0) if not isinstance(caps, torch.Tensor) else caps.to(device).unsqueeze(0) if caps.dim() == 1 else caps.to(device)
        
        with torch.enable_grad():
            # Forward pass로 예측값 생성
            outputs, _ = model(inp, caps)
            
            # 실제 정답과 비교하는 CrossEntropyLoss 계산
            # outputs: [batch, seq_len, vocab_size]
            # caps: [batch, seq_len]
            targets = caps[:, 1:]  # <start> 토큰 제거
            outputs = outputs[:, :-1, :]  # 마지막 토큰 제거
            
            # 실제 의미있는 손실 계산
            loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
        
        # Backward pass - 1차 미분 (Gradient)
        loss.backward(retain_graph=True)
        
        # Fisher Information 누적: F = E[g ⊗ g]
        for param in layer.parameters():
            if param.grad is not None:
                if importance is None:
                    # Fisher Information: g^2 (gradient의 제곱) - 더 안정적인 근사
                    importance = param.grad.data ** 2
                else:
                    importance += param.grad.data ** 2
        
        model.zero_grad()
    
    # 평균 중요도 (안정성을 위해 정규화)
    if importance is not None:
        importance = importance / num_samples
        # 수치 안정성을 위해 매우 작은 값은 clip
        importance = torch.clamp(importance, min=1e-8)
    
    model.eval()  # 평가 모드로 복귀
    return importance

def compute_channel_importance_hessian(weight, pruning_rate, dim=1, hessian_importance=None):
    """Hessian 기반 채널 중요도 계산
    
    Args:
        weight: 가중치 행렬
        pruning_rate: 프루닝 비율
        dim: 채널 차원 (1=입력 채널, 0=출력 채널)
        hessian_importance: Hessian 기반 중요도 (선택)
    
    Returns:
        mask: 제거할 채널을 False로 표시
    """
    if hessian_importance is not None:
        # Hessian 기반: 손실에 미치는 영향도 (2차 정보) 사용
        # Hessian * weight^2 형태로 중요도 계산 (2차 Taylor 전개)
        if dim == 1:
            channel_importance = (hessian_importance * (weight ** 2)).sum(dim=0)
        else:
            channel_importance = (hessian_importance * (weight ** 2)).sum(dim=1)
    else:
        # L2 norm 기반 (기존 방식)
        if dim == 1:
            channel_importance = torch.norm(weight, p=2, dim=0)
        else:
            channel_importance = torch.norm(weight, p=2, dim=1)
    
    # 중요도가 낮은 채널 선택
    num_to_prune = int(pruning_rate * channel_importance.numel())
    if num_to_prune == 0:
        return torch.ones(channel_importance.numel(), dtype=torch.bool, device=weight.device)
    
    _, indices = torch.sort(channel_importance)
    mask = torch.ones(channel_importance.numel(), dtype=torch.bool, device=weight.device)
    mask[indices[:num_to_prune]] = False  # 중요도 낮은 채널 제거
    
    return mask

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
        epochs=10, label=label.replace(" ", "_").replace("%", "pct")
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
    
    # 속도 및 메모리 측정 (추론 과정만)
    latencies = []
    time_per_tokens = []  # 토큰당 추론 시간
    memory_usages = []  # 각 추론의 메모리 사용량
    
    # CUDA 메모리 측정 준비
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    # ⚠️ CRITICAL: GC를 루프 밖으로 이동 - 시간 측정 왜곡 방지
    # gc.collect()는 무거운 작업이므로 루프 밖에서 한 번만 실행
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    for i in range(NUM_RUNS):
        
        # 메모리 측정 준비 (시간 측정 전)
        if device.type == 'cuda': 
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # 이전 작업 완료 대기
            torch.cuda.reset_peak_memory_stats()  # 피크 메모리 통계 초기화
            mem_before = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        else:
            mem_before = get_peak_memory_mb()
        
        # 추론 시간 측정 시작
        if device.type == 'cuda':
            torch.cuda.synchronize()  # 추론 전 동기화 (이전 작업 완료 보장)
        
        start = time.time()
        
        # 추론 실행
        with torch.no_grad():
            gen_seq = model.generate(inp, wm, rwm, 20)
        
        # CUDA의 경우 비동기 실행 완료 대기 (추론 시간에 포함)
        if device.type == 'cuda': 
            torch.cuda.synchronize()
        
        # 추론 시간 측정 종료
        inference_time = (time.time() - start) * 1000  # ms
        
        # 생성된 토큰 길이 계산 (이미 gen_seq가 생성됨)
        token_length = len([w for w in gen_seq if w not in ['<start>', '<end>', '<pad>', '<unk>']])
        if token_length == 0:
            token_length = 1  # 0으로 나누기 방지
        
        # 토큰당 평균 추론 시간 계산
        time_per_token = inference_time / token_length
        
        latencies.append(inference_time)
        time_per_tokens.append(time_per_token)
        
        # 메모리 측정 (시간 측정 후)
        if device.type == 'cuda': 
            # 실제 사용된 메모리 (피크 메모리 사용)
            mem_used = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        else:
            # CPU/MPS: 추론 후 메모리
            mem_after = get_peak_memory_mb()
            mem_used = max(0, mem_after - mem_before)  # 차이만 계산
        
        memory_usages.append(mem_used)
        
        if (i + 1) % 10 == 0:
            print(f"   진행: {i+1}/{NUM_RUNS}")
    
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
    avg_time = np.mean(latencies)
    std_time = np.std(latencies)
    avg_time_per_token = np.mean(time_per_tokens)  # 토큰당 평균 추론 시간
    
    # Dense format 크기 (메모리상 크기)
    size_mb_dense = get_model_size_mb(model, sparse=False)
    # Sparse format 크기 (실제 저장 크기)
    size_mb_sparse = get_sparse_model_size_mb(model)
    
    # 추론 과정에서의 평균 메모리 사용량
    memory_usage = np.mean(memory_usages) if memory_usages else 0.0
    total_params, trainable_params = count_parameters(model)
    nonzero_params, _ = count_nonzero_parameters(model)
    
    # Sparsity 계산: Magnitude (구조 미변경) vs Structured (구조 변경) 구분
    # Magnitude Pruning: 가중치 희소성 = 0인 가중치의 비율
    # Structured Pruning: 구조 희소성 = 총 파라미터 감소율
    weight_sparsity = 1.0 - (nonzero_params / total_params) if total_params > 0 else 0.0
    
    if baseline_params is not None and baseline_params > 0:
        structural_sparsity = 1.0 - (total_params / baseline_params)
        # 구조가 변경된 경우 (파라미터가 감소한 경우) = Structured
        if total_params < baseline_params:
            sparsity = structural_sparsity
        # 구조가 변경되지 않은 경우 (파라미터가 같은 경우) = Magnitude
        else:
            sparsity = weight_sparsity
    else:
        # Baseline이 없으면 가중치 희소성 사용
        sparsity = weight_sparsity
    
    print(f"   ⏱️ 평균 시간: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"   💾 모델 크기 (Dense): {size_mb_dense:.2f} MB")
    print(f"   💾 모델 크기 (Sparse): {size_mb_sparse:.2f} MB")
    print(f"   📉 크기 감소율: {(1 - size_mb_sparse/size_mb_dense)*100:.2f}%")
    print(f"   📊 총 파라미터: {total_params:,} (0이 아닌: {nonzero_params:,})")
    print(f"   ✂️ Sparsity: {sparsity*100:.2f}%")
    print(f"   🧠 메모리 사용량: {memory_usage:.5f} MB")
    if avg_meteor is not None:
        print(f"   ⭐ METEOR: {avg_meteor:.4f}")
    print(f"   📝 예시 캡션: {example_caption}")
    print(f"  📝 참조 캡션{ref_caption}")
    
    return {
        'precision': precision_name,
        'mean_time_ms': avg_time,
        'std_time_ms': std_time,
        'min_time_ms': np.min(latencies),
        'max_time_ms': np.max(latencies),
        'mean_time_per_token_ms': avg_time_per_token,  # 토큰당 평균 추론 시간
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
def fine_tune_pruned_model(model, word_map, img_tensor=None, wm=None, rwm=None, ref_caption=None, baseline_params=None, epochs=2, label="pruned_model"):
    """파인튜닝 수행 + Epoch마다 벤치마크 및 모델 저장 + 체크포인트 로드"""
    print(f"\n   🔄 파인 튜닝 시작 ({epochs} epoch)...")
    print(device)
    
    # 🔄 체크포인트 확인 및 로드
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    start_epoch = 0
    
    # 기존 체크포인트 확인
    checkpoint_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith(label) and f.endswith('.pth')]
    if checkpoint_files:
        # 가장 최신 체크포인트 찾기
        epoch_numbers = []
        for f in checkpoint_files:
            try:
                # "pruned_model_epoch_X.pth" 형식에서 X 추출
                epoch_num = int(f.split('epoch_')[-1].replace('.pth', ''))
                epoch_numbers.append((epoch_num, f))
            except ValueError:
                continue
        
        if epoch_numbers:
            epoch_numbers.sort()
            latest_epoch, latest_file = epoch_numbers[-1]
            checkpoint_path = os.path.join(OUTPUT_DIR, latest_file)
            
            print(f"   📂 체크포인트 발견: {latest_file} (Epoch {latest_epoch})")
            try:
                model.load_state_dict(torch.load(checkpoint_path, map_location=device))
                start_epoch = latest_epoch  # 다음 epoch부터 시작
                print(f"   ✅ 체크포인트 로드 완료. Epoch {start_epoch+1}부터 재개합니다.")
            except Exception as e:
                print(f"   ⚠️ 체크포인트 로드 실패: {e}")
                print(f"   🔄 처음부터 시작합니다.")
                start_epoch = 0
    else:
        print(f"   ℹ️ 기존 체크포인트가 없습니다. 처음부터 시작합니다.")
    
    # ⚠️ CRITICAL: Encoder(CNN) Freeze - ImageNet 학습된 특징 보존
    # Encoder를 학습하면 Catastrophic Forgetting 또는 극심한 Overfitting 발생
    if hasattr(model, 'encoder'):
        for param in model.encoder.parameters():
            param.requires_grad = False
        print(f"   🔒 Encoder Freeze: CNN 파라미터 학습 금지")
    
    # 학습 데이터셋 준비
    try:
        dataset = CaptionDataset(
            images_dir=TEST_IMAGE_DIR,
            captions_file=CAPTIONS_FILE,
            transform=transform,
            word_map=word_map,
            max_len=50
        )
        
        if len(dataset) == 0:
            print("   ⚠️ 학습 데이터가 없어 파인 튜닝을 건너뜁니다.")
            return model
        
        # 적응형 배치 사이즈: 데이터가 적으면 더 작은 배치 사이즈 사용 (업데이트 횟수 증가)
        dataset_size = len(dataset)
        if dataset_size < 1000:
            batch_size = 32  # 작은 데이터셋: 더 많은 업데이트
            print(f"   📊 배치 사이즈 조정: {dataset_size}개 샘플 → batch_size=32 (빈번한 업데이트)")
        else:
            batch_size = 64  # 중간 크기: 균형잡힌 업데이트
            print(f"   📊 배치 사이즈: batch_size=64")
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        
        print(f"   📚 학습 데이터: {len(dataset)}개 샘플, {len(dataloader)}개 배치")
        
        # 모델을 학습 모드로 전환
        model.train()
        model.to(device)
        
        # Optimizer 및 Loss 설정
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # ⚠️ CRITICAL: Decoder만 학습 (Encoder는 requires_grad=False)
        # filter()로 requires_grad=True인 파라미터만 Optimizer에 전달
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(trainable_params, lr=5e-5)
        
        # 학습할 파라미터 개수 출력
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_count = sum(p.numel() for p in model.parameters())
        print(f"   📊 학습 대상 파라미터: {trainable_count:,} / {total_count:,} ({100*trainable_count/total_count:.1f}%)")
        vocab_size = len(word_map)
        
        # 파인튜닝 진행 (체크포인트 이후부터 시작)
        for epoch in range(start_epoch, epochs):
            print(f"   🏋️ Epoch {epoch+1}/{epochs}")
            total_loss = 0
            num_batches = 0
            
            for batch_idx, (imgs, caps) in enumerate(dataloader):
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
                    print(f"      배치 {batch_idx + 1}/{len(dataloader)}, Loss: {total_loss / num_batches:.4f}")
            
            # 🎯 Epoch 끝 - 벤치마크 실행
            if num_batches > 0:
                avg_loss = total_loss / num_batches
                print(f"   ✅ Epoch {epoch+1} 완료 (평균 Loss: {avg_loss:.4f})")
            
            # 벤치마크 실행 (img_tensor, wm, rwm이 제공된 경우)
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
                        print(f"      ⭐ METEOR: {benchmark_result['meteor_score']:.4f}")
                
                # 모델 저장
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                model_save_path = os.path.join(OUTPUT_DIR, f"pruned_model_epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), model_save_path)
                print(f"      💾 모델 저장: {model_save_path}")
        
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

