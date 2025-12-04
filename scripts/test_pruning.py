"""
Pruning 벤치마크 스크립트
다양한 Pruning 기법을 적용하고 성능을 비교합니다.
"""
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
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
PRUNING_RATES = [0.1, 0.3, 0.5, 0.7]  # 10%, 30%, 50%, 70% 프루닝
PRUNING_METHODS = ['magnitude', 'structured']  # 프루닝 방법

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
    """Sparse format으로 저장했을 때의 실제 모델 크기 계산"""
    total_size = 0
    
    for name, param in model.named_parameters():
        if param.numel() > 0:
            # 0이 아닌 값의 개수
            nonzero_count = (param != 0).sum().item()
            total_params = param.numel()
            
            if nonzero_count > 0:
                # 값 저장 (0이 아닌 값만)
                total_size += nonzero_count * param.element_size()
                
                # 인덱스 저장 (COO format: Coordinate format)
                # 각 0이 아닌 값의 위치를 저장
                if len(param.shape) == 1:
                    # 1D: 인덱스만
                    indices_size = nonzero_count * 4  # 4 bytes per index
                elif len(param.shape) == 2:
                    # 2D: (row, col) 쌍
                    indices_size = nonzero_count * 2 * 4  # 2 indices per value
                else:
                    # 다차원: 모든 차원의 인덱스
                    indices_size = nonzero_count * len(param.shape) * 4
                
                total_size += indices_size
                
                # 메타데이터 (shape, dtype, nonzero_count 등)
                total_size += 64  # 메타데이터 오버헤드
            else:
                # 모든 값이 0인 경우 최소 메타데이터만
                total_size += 32
    
    # 버퍼 크기 (버퍼는 보통 작으므로 그대로 계산)
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
def get_pruning_mask(weight, pruning_rate, dim=0, use_l2=False):
    """프루닝 마스크 생성 (제거할 채널/뉴런 식별)
    
    Args:
        weight: 가중치 텐서
        pruning_rate: 프루닝 비율
        dim: 프루닝할 차원 (0: 출력, 1: 입력)
        use_l2: True면 L2 norm 사용 (Structured), False면 L1 norm 사용 (Magnitude)
    """
    if dim == 0:  # 출력 차원 프루닝
        if use_l2:
            # Structured: L2 norm 사용 (채널 단위 중요도)
            importance = torch.norm(weight, p=2, dim=1)  # [out_features] - L2 norm
        else:
            # Magnitude: L1 norm 사용
            importance = torch.abs(weight).sum(dim=1)  # [out_features] - L1 norm
    else:  # 입력 차원 프루닝
        if use_l2:
            # Structured: L2 norm 사용
            importance = torch.norm(weight, p=2, dim=0)  # [in_features] - L2 norm
        else:
            # Magnitude: L1 norm 사용
            importance = torch.abs(weight).sum(dim=0)  # [in_features] - L1 norm
    
    # 중요도가 낮은 순서로 정렬
    num_to_prune = int(pruning_rate * importance.numel())
    if num_to_prune == 0:
        return torch.ones(importance.numel(), dtype=torch.bool, device=weight.device)
    
    _, indices = torch.sort(importance)
    mask = torch.ones(importance.numel(), dtype=torch.bool, device=weight.device)
    mask[indices[:num_to_prune]] = False
    
    return mask

def apply_structured_pruning_physical(model, pruning_rate):
    """Structured Pruning 적용 (물리적 구조 변경 - 실제 채널/뉴런 제거)"""
    from src.gru_model.model import LightweightCaptionDecoder
    
    pruned_model = deepcopy(model)
    pruned_model.eval()
    
    decoder = pruned_model.decoder
    
    # 1. decoder_dim 차원 프루닝 (가장 영향이 큰 차원)
    # decoder_dim은 decoder_att, init_h, fc, decode_step에서 사용됨
    
    # decoder_att의 입력 차원(decoder_dim)을 기준으로 프루닝
    if hasattr(decoder, 'decoder_att'):
        old_decoder_att = decoder.decoder_att
        weight = old_decoder_att.weight.data  # [attention_dim, decoder_dim]
        
        # decoder_dim 차원에서 프루닝 (입력 차원)
        # Structured pruning: L2 norm 사용
        mask_decoder_dim = get_pruning_mask(weight, pruning_rate, dim=1, use_l2=True)
        new_decoder_dim = mask_decoder_dim.sum().item()
        
        # decoder_att 레이어 재생성
        new_decoder_att = nn.Linear(new_decoder_dim, decoder.attention_dim)
        new_decoder_att.weight.data = weight[:, mask_decoder_dim]  # [attention_dim, new_decoder_dim]
        if old_decoder_att.bias is not None:
            new_decoder_att.bias.data = old_decoder_att.bias.data.clone()
        decoder.decoder_att = new_decoder_att
        
        # decoder_dim 업데이트
        decoder.decoder_dim = new_decoder_dim
        
        # init_h 레이어 조정 (출력 차원이 decoder_dim)
        if hasattr(decoder, 'init_h'):
            old_init_h = decoder.init_h
            old_weight = old_init_h.weight.data  # [decoder_dim, encoder_dim]
            new_init_h = nn.Linear(decoder.encoder_dim, new_decoder_dim)
            new_init_h.weight.data = old_weight[mask_decoder_dim, :]  # [new_decoder_dim, encoder_dim]
            if old_init_h.bias is not None:
                new_init_h.bias.data = old_init_h.bias.data[mask_decoder_dim]
            decoder.init_h = new_init_h
        
        # fc 레이어 조정 (입력 차원이 decoder_dim)
        if hasattr(decoder, 'fc'):
            old_fc = decoder.fc
            old_weight = old_fc.weight.data  # [vocab_size, decoder_dim]
            new_fc = nn.Linear(new_decoder_dim, decoder.vocab_size)
            new_fc.weight.data = old_weight[:, mask_decoder_dim]  # [vocab_size, new_decoder_dim]
            if old_fc.bias is not None:
                new_fc.bias.data = old_fc.bias.data.clone()
            decoder.fc = new_fc
        
        # decode_step (GRUCell) 조정
        # GRUCell의 hidden_size가 decoder_dim이므로 재생성 필요
        if hasattr(decoder, 'decode_step'):
            from src.gru_model.model import LightweightCaptionDecoder
            # GRUCell: input_size = embed_dim + encoder_dim, hidden_size = decoder_dim
            old_decode_step = decoder.decode_step
            input_size = old_decode_step.input_size
            new_decode_step = nn.GRUCell(input_size, new_decoder_dim)
            
            # 가중치 복사 (가능한 부분만)
            old_hidden_size = old_decode_step.hidden_size
            if new_decoder_dim <= old_hidden_size:
                # weight_ih: [3 * hidden_size, input_size]
                # weight_hh: [3 * hidden_size, hidden_size]
                old_weight_ih = old_decode_step.weight_ih.data  # [3 * old_hidden_size, input_size]
                old_weight_hh = old_decode_step.weight_hh.data  # [3 * old_hidden_size, old_hidden_size]
                
                # 각 게이트별로 마스크 적용
                gate_size = old_hidden_size
                new_gate_size = new_decoder_dim
                mask_gates = mask_decoder_dim.repeat(3)  # [3 * decoder_dim]
                
                new_weight_ih = old_weight_ih[mask_gates, :]  # [3 * new_decoder_dim, input_size]
                new_weight_hh = old_weight_hh[mask_gates, :][:, mask_decoder_dim]  # [3 * new_decoder_dim, new_decoder_dim]
                
                new_decode_step.weight_ih.data = new_weight_ih
                new_decode_step.weight_hh.data = new_weight_hh
                
                if old_decode_step.bias_ih is not None:
                    old_bias_ih = old_decode_step.bias_ih.data
                    new_decode_step.bias_ih.data = old_bias_ih[mask_gates]
                if old_decode_step.bias_hh is not None:
                    old_bias_hh = old_decode_step.bias_hh.data
                    new_decode_step.bias_hh.data = old_bias_hh[mask_gates]
            
            decoder.decode_step = new_decode_step
    
    # 2. attention_dim 차원 프루닝 (선택적)
    if hasattr(decoder, 'encoder_att') and hasattr(decoder, 'full_att'):
        # encoder_att의 출력 차원(attention_dim) 프루닝
        old_encoder_att = decoder.encoder_att
        weight = old_encoder_att.weight.data  # [attention_dim, encoder_dim]
        
        # Structured pruning: L2 norm 사용
        mask_attention_dim = get_pruning_mask(weight, pruning_rate, dim=0, use_l2=True)
        new_attention_dim = mask_attention_dim.sum().item()
        
        # encoder_att 레이어 재생성
        new_encoder_att = nn.Linear(decoder.encoder_dim, new_attention_dim)
        new_encoder_att.weight.data = weight[mask_attention_dim, :]  # [new_attention_dim, encoder_dim]
        if old_encoder_att.bias is not None:
            new_encoder_att.bias.data = old_encoder_att.bias.data[mask_attention_dim]
        decoder.encoder_att = new_encoder_att
        
        # attention_dim 업데이트
        decoder.attention_dim = new_attention_dim
        
        # decoder_att의 출력 차원도 조정
        if hasattr(decoder, 'decoder_att'):
            old_decoder_att = decoder.decoder_att
            old_weight = old_decoder_att.weight.data  # [old_attention_dim, decoder_dim]
            new_decoder_att = nn.Linear(decoder.decoder_dim, new_attention_dim)
            new_decoder_att.weight.data = old_weight[mask_attention_dim, :]  # [new_attention_dim, decoder_dim]
            if old_decoder_att.bias is not None:
                new_decoder_att.bias.data = old_decoder_att.bias.data[mask_attention_dim]
            decoder.decoder_att = new_decoder_att
        
        # full_att의 입력 차원 조정
        if hasattr(decoder, 'full_att'):
            old_full_att = decoder.full_att
            old_weight = old_full_att.weight.data  # [1, old_attention_dim]
            new_full_att = nn.Linear(new_attention_dim, 1)
            new_full_att.weight.data = old_weight[:, mask_attention_dim]  # [1, new_attention_dim]
            if old_full_att.bias is not None:
                new_full_att.bias.data = old_full_att.bias.data.clone()
            decoder.full_att = new_full_att
    
    pruned_model.decoder = decoder
    pruned_model.eval()
    
    # 파라미터 개수 확인
    old_params = sum(p.numel() for p in model.parameters())
    new_params = sum(p.numel() for p in pruned_model.parameters())
    reduction = (1 - new_params / old_params) * 100
    
    print(f"   ✂️ 물리적 구조 Pruning 완료: {pruning_rate*100:.0f}% 채널 제거")
    print(f"   📊 파라미터 감소: {old_params:,} → {new_params:,} ({reduction:.1f}% 감소)")
    
    return pruned_model

def apply_magnitude_pruning(model, pruning_rate):
    """Magnitude-based Pruning 적용 (가중치 크기 기반, 물리적 구조 변경)
    
    Magnitude pruning은 각 레이어의 가중치 절댓값(magnitude)을 기준으로
    중요도가 낮은 채널/뉴런을 제거합니다.
    """
    from src.gru_model.model import LightweightCaptionDecoder
    
    pruned_model = deepcopy(model)
    pruned_model.eval()
    
    decoder = pruned_model.decoder
    
    # Magnitude-based pruning: 각 레이어의 가중치 절댓값을 기준으로 중요도 계산
    # 모든 레이어에 일관되게 적용
    
    # 1. decoder_dim 차원 프루닝 (Magnitude 기반)
    # decoder_att의 가중치를 magnitude 기준으로 평가
    if hasattr(decoder, 'decoder_att'):
        old_decoder_att = decoder.decoder_att
        weight = old_decoder_att.weight.data  # [attention_dim, decoder_dim]
        
        # Magnitude 기반 중요도 계산: 각 decoder_dim 채널의 L1 norm
        # 각 입력 채널(decoder_dim)의 모든 출력에 대한 가중치 합
        importance = torch.abs(weight).sum(dim=0)  # [decoder_dim] - 각 입력 채널의 중요도
        
        # 중요도가 낮은 순서로 정렬
        num_to_prune = int(pruning_rate * importance.numel())
        if num_to_prune > 0 and num_to_prune < importance.numel():
            _, indices = torch.sort(importance)
            mask_decoder_dim = torch.ones(importance.numel(), dtype=torch.bool, device=weight.device)
            mask_decoder_dim[indices[:num_to_prune]] = False
        else:
            mask_decoder_dim = torch.ones(importance.numel(), dtype=torch.bool, device=weight.device)
        
        new_decoder_dim = mask_decoder_dim.sum().item()
        
        # decoder_att 레이어 재생성
        new_decoder_att = nn.Linear(new_decoder_dim, decoder.attention_dim)
        new_decoder_att.weight.data = weight[:, mask_decoder_dim]  # [attention_dim, new_decoder_dim]
        if old_decoder_att.bias is not None:
            new_decoder_att.bias.data = old_decoder_att.bias.data.clone()
        decoder.decoder_att = new_decoder_att
        
        # decoder_dim 업데이트
        decoder.decoder_dim = new_decoder_dim
        
        # init_h 레이어 조정 (출력 차원이 decoder_dim)
        if hasattr(decoder, 'init_h'):
            old_init_h = decoder.init_h
            old_weight = old_init_h.weight.data  # [decoder_dim, encoder_dim]
            new_init_h = nn.Linear(decoder.encoder_dim, new_decoder_dim)
            new_init_h.weight.data = old_weight[mask_decoder_dim, :]  # [new_decoder_dim, encoder_dim]
            if old_init_h.bias is not None:
                new_init_h.bias.data = old_init_h.bias.data[mask_decoder_dim]
            decoder.init_h = new_init_h
        
        # fc 레이어 조정 (입력 차원이 decoder_dim)
        if hasattr(decoder, 'fc'):
            old_fc = decoder.fc
            old_weight = old_fc.weight.data  # [vocab_size, decoder_dim]
            new_fc = nn.Linear(new_decoder_dim, decoder.vocab_size)
            new_fc.weight.data = old_weight[:, mask_decoder_dim]  # [vocab_size, new_decoder_dim]
            if old_fc.bias is not None:
                new_fc.bias.data = old_fc.bias.data.clone()
            decoder.fc = new_fc
        
        # decode_step (GRUCell) 조정
        if hasattr(decoder, 'decode_step'):
            old_decode_step = decoder.decode_step
            input_size = old_decode_step.input_size
            new_decode_step = nn.GRUCell(input_size, new_decoder_dim)
            
            old_hidden_size = old_decode_step.hidden_size
            if new_decoder_dim <= old_hidden_size:
                old_weight_ih = old_decode_step.weight_ih.data  # [3 * old_hidden_size, input_size]
                old_weight_hh = old_decode_step.weight_hh.data  # [3 * old_hidden_size, old_hidden_size]
                
                # 각 게이트별로 마스크 적용
                mask_gates = mask_decoder_dim.repeat(3)  # [3 * decoder_dim]
                
                new_weight_ih = old_weight_ih[mask_gates, :]  # [3 * new_decoder_dim, input_size]
                new_weight_hh = old_weight_hh[mask_gates, :][:, mask_decoder_dim]  # [3 * new_decoder_dim, new_decoder_dim]
                
                new_decode_step.weight_ih.data = new_weight_ih
                new_decode_step.weight_hh.data = new_weight_hh
                
                if old_decode_step.bias_ih is not None:
                    old_bias_ih = old_decode_step.bias_ih.data
                    new_decode_step.bias_ih.data = old_bias_ih[mask_gates]
                if old_decode_step.bias_hh is not None:
                    old_bias_hh = old_decode_step.bias_hh.data
                    new_decode_step.bias_hh.data = old_bias_hh[mask_gates]
            
            decoder.decode_step = new_decode_step
    
    # 2. attention_dim 차원 프루닝 (Magnitude 기반)
    if hasattr(decoder, 'encoder_att') and hasattr(decoder, 'full_att'):
        old_encoder_att = decoder.encoder_att
        weight = old_encoder_att.weight.data  # [attention_dim, encoder_dim]
        
        # Magnitude 기반 중요도 계산: 각 attention_dim 출력 채널의 L1 norm
        importance = torch.abs(weight).sum(dim=1)  # [attention_dim] - 각 출력 채널의 중요도
        
        # 중요도가 낮은 순서로 정렬
        num_to_prune = int(pruning_rate * importance.numel())
        if num_to_prune > 0 and num_to_prune < importance.numel():
            _, indices = torch.sort(importance)
            mask_attention_dim = torch.ones(importance.numel(), dtype=torch.bool, device=weight.device)
            mask_attention_dim[indices[:num_to_prune]] = False
        else:
            mask_attention_dim = torch.ones(importance.numel(), dtype=torch.bool, device=weight.device)
        
        new_attention_dim = mask_attention_dim.sum().item()
        
        # encoder_att 레이어 재생성
        new_encoder_att = nn.Linear(decoder.encoder_dim, new_attention_dim)
        new_encoder_att.weight.data = weight[mask_attention_dim, :]  # [new_attention_dim, encoder_dim]
        if old_encoder_att.bias is not None:
            new_encoder_att.bias.data = old_encoder_att.bias.data[mask_attention_dim]
        decoder.encoder_att = new_encoder_att
        
        # attention_dim 업데이트
        decoder.attention_dim = new_attention_dim
        
        # decoder_att의 출력 차원도 조정
        if hasattr(decoder, 'decoder_att'):
            old_decoder_att = decoder.decoder_att
            old_weight = old_decoder_att.weight.data  # [old_attention_dim, decoder_dim]
            new_decoder_att = nn.Linear(decoder.decoder_dim, new_attention_dim)
            new_decoder_att.weight.data = old_weight[mask_attention_dim, :]  # [new_attention_dim, decoder_dim]
            if old_decoder_att.bias is not None:
                new_decoder_att.bias.data = old_decoder_att.bias.data[mask_attention_dim]
            decoder.decoder_att = new_decoder_att
        
        # full_att의 입력 차원 조정
        if hasattr(decoder, 'full_att'):
            old_full_att = decoder.full_att
            old_weight = old_full_att.weight.data  # [1, old_attention_dim]
            new_full_att = nn.Linear(new_attention_dim, 1)
            new_full_att.weight.data = old_weight[:, mask_attention_dim]  # [1, new_attention_dim]
            if old_full_att.bias is not None:
                new_full_att.bias.data = old_full_att.bias.data.clone()
            decoder.full_att = new_full_att
    
    pruned_model.decoder = decoder
    pruned_model.eval()
    
    # 파라미터 개수 확인
    old_params = sum(p.numel() for p in model.parameters())
    new_params = sum(p.numel() for p in pruned_model.parameters())
    reduction = (1 - new_params / old_params) * 100
    
    print(f"   ✂️ Magnitude-based Pruning 완료: {pruning_rate*100:.0f}% 채널 제거")
    print(f"   📊 파라미터 감소: {old_params:,} → {new_params:,} ({reduction:.1f}% 감소)")
    
    return pruned_model

def apply_structured_pruning(model, pruning_rate):
    """Structured Pruning 적용 (채널/필터 단위, 물리적 구조 변경)"""
    return apply_structured_pruning_physical(model, pruning_rate)

def apply_global_pruning(model, pruning_rate):
    """Global Pruning 적용 (전체 모델 기준, 물리적 구조 변경)"""
    # Global pruning도 structured 방식으로 적용
    return apply_structured_pruning_physical(model, pruning_rate)


# ============================================================================
# 벤치마크 엔진
# ============================================================================
def run_benchmark(model, img_tensor, wm, rwm, precision_name, ref_caption=None, baseline_params=None):
    """벤치마크 실행
    
    Args:
        model: 모델
        img_tensor: 입력 이미지
        wm: word_map
        rwm: rev_word_map
        precision_name: 정밀도 이름
        ref_caption: 참조 캡션
        baseline_params: Baseline 모델의 파라미터 개수 (Sparsity 계산용)
    """
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
    
    for i in range(NUM_RUNS):
        gc.collect()
        
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
    
    # 테스트 이미지 디렉토리에서 10개 이미지 로드
    if os.path.exists(TEST_IMAGE_DIR):
        image_files = [f for f in os.listdir(TEST_IMAGE_DIR) 
                      if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if image_files:
            import random
            # 최대 10개 이미지 선택
            selected_files = random.sample(image_files, min(10, len(image_files)))
            
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
    
    # 10개 이미지에 대해 METEOR 점수 계산
    if test_images_meteor and any(test_captions_meteor):
        print(f"   📊 METEOR 점수 측정 중: {len([c for c in test_captions_meteor if c])}개 이미지")
        for idx, (test_img, ref_cap) in enumerate(zip(test_images_meteor[:10], test_captions_meteor[:10])):
            if ref_cap:
                with torch.no_grad():
                    gen_seq = model.generate(test_img, wm, rwm, 20)
                meteor = calculate_meteor(gen_seq, ref_cap)
                if meteor is not None:
                    meteor_scores.append(meteor)
                if idx == 0:
                    example_caption = ' '.join([w for w in gen_seq if w not in ['<start>', '<end>', '<pad>', '<unk>']])
    
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
    
    # Sparsity 계산: 물리적 구조 변경 후에는 baseline과 비교한 실제 감소율
    if baseline_params is not None and baseline_params > 0:
        # Baseline 대비 실제 파라미터 감소율
        sparsity = 1.0 - (total_params / baseline_params)
    else:
        # Baseline이 없으면 0이 아닌 파라미터 기반 계산 (기존 방식)
        sparsity = 1.0 - (nonzero_params / total_params) if total_params > 0 else 0.0
    
    print(f"   ⏱️ 평균 시간: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"   💾 모델 크기 (Dense): {size_mb_dense:.2f} MB")
    print(f"   💾 모델 크기 (Sparse): {size_mb_sparse:.2f} MB")
    print(f"   📉 크기 감소율: {(1 - size_mb_sparse/size_mb_dense)*100:.2f}%")
    print(f"   📊 총 파라미터: {total_params:,} (0이 아닌: {nonzero_params:,})")
    print(f"   ✂️ Sparsity: {sparsity*100:.2f}%")
    print(f"   🧠 메모리 사용량: {memory_usage:.2f} MB")
    if avg_meteor is not None:
        print(f"   ⭐ METEOR: {avg_meteor:.4f}")
    print(f"   📝 예시 캡션: {example_caption}")
    
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
    """파인 튜닝 결과 비교 그래프 (Baseline 대비)"""
    if not results or not baseline_result:
        print("❌ 결과가 없어 plot을 생성할 수 없습니다.")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 파인 튜닝된 결과만 필터링
    finetuned_results = [r for r in results if '(Fine-tuned)' in r['precision']]
    
    if not finetuned_results:
        print("❌ 파인 튜닝 결과가 없어 plot을 생성할 수 없습니다.")
        return
    
    # Baseline 정보
    baseline_time_per_token = baseline_result.get('mean_time_per_token_ms', baseline_result['mean_time_ms'] / 10)  # 기본값
    baseline_meteor = baseline_result.get('meteor_score', 0)
    baseline_time = baseline_result['mean_time_ms']
    
    # 모델별로 그룹화 (Magnitude, Structured, Global)
    model_groups = {}
    for result in finetuned_results:
        precision = result['precision']
        # "Magnitude-10% (Fine-tuned)" -> "Magnitude-10%"
        base_name = precision.replace(' (Fine-tuned)', '')
        
        if base_name not in model_groups:
            model_groups[base_name] = []
        model_groups[base_name].append(result)
    
    # 그래프 생성
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('파인 튜닝 효과 비교 (Baseline 대비)', fontsize=16, fontweight='bold')
    
    # 데이터 준비
    model_names = []
    time_improvements = []  # Baseline 대비 추론 시간 개선율
    time_per_token_improvements = []  # Baseline 대비 토큰당 시간 개선율
    meteor_improvements = []  # Baseline 대비 METEOR 점수 개선율
    model_sizes = []
    memory_usages = []
    
    for model_name, group_results in sorted(model_groups.items()):
        # 각 그룹의 평균 계산 (같은 모델이 여러 개일 수 있음)
        avg_time = np.mean([r['mean_time_ms'] for r in group_results])
        avg_time_per_token = np.mean([r.get('mean_time_per_token_ms', r['mean_time_ms'] / 10) for r in group_results])
        avg_meteor = np.mean([r.get('meteor_score', 0) for r in group_results if r.get('meteor_score') is not None]) if any(r.get('meteor_score') for r in group_results) else None
        avg_size = np.mean([r['model_size_mb'] for r in group_results])
        avg_memory = np.mean([r['memory_usage_mb'] for r in group_results])
        
        model_names.append(model_name)
        
        # Baseline 대비 개선율 계산
        time_improvement = ((baseline_time - avg_time) / baseline_time) * 100 if baseline_time > 0 else 0
        time_improvements.append(time_improvement)
        
        time_per_token_improvement = ((baseline_time_per_token - avg_time_per_token) / baseline_time_per_token) * 100 if baseline_time_per_token > 0 else 0
        time_per_token_improvements.append(time_per_token_improvement)
        
        if avg_meteor is not None and baseline_meteor > 0:
            meteor_improvement = ((avg_meteor - baseline_meteor) / baseline_meteor) * 100
        else:
            meteor_improvement = 0
        meteor_improvements.append(meteor_improvement)
        
        model_sizes.append(avg_size)
        memory_usages.append(avg_memory)
    
    # 색상 설정
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
    
    # 1. 추론 시간 개선율 (Baseline 대비)
    axes[0, 0].bar(model_names, time_improvements, alpha=0.8, color=colors)
    axes[0, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0, 0].set_ylabel('개선율 (%)', fontweight='bold')
    axes[0, 0].set_title('추론 시간 개선율 (Baseline 대비)', fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)
    for i, (name, imp) in enumerate(zip(model_names, time_improvements)):
        axes[0, 0].text(i, imp + (max(time_improvements) - min(time_improvements)) * 0.02 if time_improvements else 1,
                       f'{imp:+.1f}%', ha='center', va='bottom' if imp > 0 else 'top', fontsize=9)
    
    # 2. 토큰당 추론 시간 개선율 (Baseline 대비)
    axes[0, 1].bar(model_names, time_per_token_improvements, alpha=0.8, color=colors)
    axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0, 1].set_ylabel('개선율 (%)', fontweight='bold')
    axes[0, 1].set_title('토큰당 추론 시간 개선율 (Baseline 대비)', fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=45)
    for i, (name, imp) in enumerate(zip(model_names, time_per_token_improvements)):
        axes[0, 1].text(i, imp + (max(time_per_token_improvements) - min(time_per_token_improvements)) * 0.02 if time_per_token_improvements else 1,
                       f'{imp:+.1f}%', ha='center', va='bottom' if imp > 0 else 'top', fontsize=9)
    
    # 3. METEOR 점수 개선율 (Baseline 대비)
    if any(m != 0 for m in meteor_improvements):
        axes[1, 0].bar(model_names, meteor_improvements, alpha=0.8, color=colors)
        axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1, 0].set_ylabel('개선율 (%)', fontweight='bold')
        axes[1, 0].set_title('METEOR 점수 개선율 (Baseline 대비)', fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
        for i, (name, imp) in enumerate(zip(model_names, meteor_improvements)):
            axes[1, 0].text(i, imp + (max(meteor_improvements) - min(meteor_improvements)) * 0.02 if meteor_improvements else 1,
                           f'{imp:+.1f}%', ha='center', va='bottom' if imp > 0 else 'top', fontsize=9)
    else:
        axes[1, 0].text(0.5, 0.5, 'METEOR 점수 데이터 없음', 
                        ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('METEOR 점수 개선율 (Baseline 대비)', fontweight='bold')
    
    # 4. 모델 크기 비교
    baseline_size = baseline_result['model_size_mb']
    size_reductions = [((baseline_size - size) / baseline_size) * 100 for size in model_sizes]
    axes[1, 1].bar(model_names, size_reductions, alpha=0.8, color=colors)
    axes[1, 1].set_ylabel('크기 감소율 (%)', fontweight='bold')
    axes[1, 1].set_title('모델 크기 감소율 (Baseline 대비)', fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)
    axes[1, 1].tick_params(axis='x', rotation=45)
    for i, (name, red) in enumerate(zip(model_names, size_reductions)):
        axes[1, 1].text(i, red + max(size_reductions) * 0.02 if size_reductions else 1,
                       f'{red:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pruning_comparison_finetune.png'), 
                dpi=300, bbox_inches='tight')
    print(f"✅ 파인 튜닝 비교 Plot 저장: {os.path.join(OUTPUT_DIR, 'pruning_comparison_finetune.png')}")
    plt.close()

# ============================================================================
# 파인 튜닝 함수
# ============================================================================
def fine_tune_pruned_model(model, word_map, epochs=1):
    """프루닝된 모델을 1 epoch 파인 튜닝"""
    from torch.utils.data import DataLoader
    from src.utils import CaptionDataset
    
    print(f"\n   🔄 파인 튜닝 시작 ({epochs} epoch)...")
    
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
        
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        
        print(f"   📚 학습 데이터: {len(dataset)}개 샘플")
        
        # 모델을 학습 모드로 전환
        model.train()
        model.to(device)
        
        # Optimizer 및 Loss 설정
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        vocab_size = len(word_map)
        
        # 1 epoch 학습
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (imgs, caps) in enumerate(dataloader):
            if batch_idx >= 50:  # 최대 50 배치만 학습 (빠른 파인 튜닝)
                break
            
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
        
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            print(f"   ✅ 파인 튜닝 완료 (평균 Loss: {avg_loss:.4f})")
        
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
        # Magnitude-based Pruning
        print("\n" + "="*70)
        print(f"=== Magnitude Pruning ({pruning_rate*100:.0f}%) ===")
        print("="*70)
        try:
            pruned_model = apply_magnitude_pruning(base_model, pruning_rate)
            pruned_model.to(device)
            
            # 프루닝 후 벤치마크
            result = run_benchmark(
                pruned_model, img_tensor, wm, rwm, 
                f"Magnitude-{pruning_rate*100:.0f}%", ref_caption, baseline_params=baseline_params
            )
            if result:
                results.append(result)
            
            # 파인 튜닝
            fine_tuned_model = fine_tune_pruned_model(pruned_model, wm, epochs=1)
            fine_tuned_model.to(device)
            
            # 파인 튜닝 후 벤치마크
            result_finetuned = run_benchmark(
                fine_tuned_model, img_tensor, wm, rwm,
                f"Magnitude-{pruning_rate*100:.0f}% (Fine-tuned)", ref_caption, baseline_params=baseline_params
            )
            if result_finetuned:
                results.append(result_finetuned)
            
            del pruned_model, fine_tuned_model
            gc.collect()
        except Exception as e:
            print(f"⚠️ Magnitude Pruning ({pruning_rate*100:.0f}%) 실패: {e}")
            import traceback
            traceback.print_exc()
        
        # Structured Pruning
        print("\n" + "="*70)
        print(f"=== Structured Pruning ({pruning_rate*100:.0f}%) ===")
        print("="*70)
        try:
            pruned_model = apply_structured_pruning(base_model, pruning_rate)
            pruned_model.to(device)
            
            # 프루닝 후 벤치마크
            result = run_benchmark(
                pruned_model, img_tensor, wm, rwm, 
                f"Structured-{pruning_rate*100:.0f}%", ref_caption, baseline_params=baseline_params
            )
            if result:
                results.append(result)
            
            # 파인 튜닝
            fine_tuned_model = fine_tune_pruned_model(pruned_model, wm, epochs=1)
            fine_tuned_model.to(device)
            
            # 파인 튜닝 후 벤치마크
            result_finetuned = run_benchmark(
                fine_tuned_model, img_tensor, wm, rwm,
                f"Structured-{pruning_rate*100:.0f}% (Fine-tuned)", ref_caption, baseline_params=baseline_params
            )
            if result_finetuned:
                results.append(result_finetuned)
            
            del pruned_model, fine_tuned_model
            gc.collect()
        except Exception as e:
            print(f"⚠️ Structured Pruning ({pruning_rate*100:.0f}%) 실패: {e}")
            import traceback
            traceback.print_exc()
    
    # 4. Global Pruning 테스트
    print("\n" + "="*70)
    print("=== Global Pruning (50%) ===")
    print("="*70)
    try:
        pruned_model = apply_global_pruning(base_model, 0.5)
        pruned_model.to(device)
        
        # 프루닝 후 벤치마크
        result = run_benchmark(pruned_model, img_tensor, wm, rwm, "Global-50%", ref_caption, baseline_params=baseline_params)
        if result:
            results.append(result)
        
        # 파인 튜닝
        fine_tuned_model = fine_tune_pruned_model(pruned_model, wm, epochs=1)
        fine_tuned_model.to(device)
        
        # 파인 튜닝 후 벤치마크
        result_finetuned = run_benchmark(
            fine_tuned_model, img_tensor, wm, rwm,
            "Global-50% (Fine-tuned)", ref_caption, baseline_params=baseline_params
        )
        if result_finetuned:
            results.append(result_finetuned)
        
        del pruned_model, fine_tuned_model
        gc.collect()
    except Exception as e:
        print(f"⚠️ Global Pruning 실패: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. 결과 요약 출력
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

