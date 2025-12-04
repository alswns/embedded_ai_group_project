"""
Pruning 관련 유틸리티 함수들
- 파라미터 계산
- 프루닝 마스크 생성
- Hessian 중요도 계산
- 레이어 업데이트
"""

import torch
import torch.nn as nn
import numpy as np


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
    # 실제 0이 아닌 파라미터만 계산
    pass


def save_sparse_model(model, path):
    """Sparse 모델 저장"""
    try:
        # 크기 계산
        total_size = 0
        nonzero_count = 0
        
        for name, param in model.named_parameters():
            total_size += param.numel()
            nonzero_count += (param != 0).sum().item()
        
        state_dict = model.state_dict()
        torch.save(state_dict, path)
        
        return True
    except Exception as e:
        print(f"❌ 저장 실패: {e}")
        return False


def get_sparse_model_size_mb(model):
    """Sparse 모델의 메모리 크기 계산 (MB)"""
    param_size = 0
    buffer_size = 0
    
    # Sparse tensor용 추정치
    for param in model.parameters():
        if param.is_sparse:
            indices_size = param._indices().numel() * 8  # int64
            values_size = param._values().numel() * 4  # float32
            param_size += indices_size + values_size
        else:
            nonzero = (param != 0).sum().item()
            if nonzero > 0:
                param_size += nonzero * 4  # float32
    
    for buffer in model.buffers():
        if buffer.is_sparse:
            indices_size = buffer._indices().numel() * 8
            values_size = buffer._values().numel() * 4
            buffer_size += indices_size + values_size
        else:
            buffer_size += buffer.numel() * 4
    
    return (param_size + buffer_size) / (1024 * 1024)


def get_pruning_mask(weight, pruning_rate, dim=0, use_l2=True):
    """
    L2 norm 또는 magnitude 기반 프루닝 마스크 생성
    
    Args:
        weight: 프루닝할 가중치 텐서
        pruning_rate: 프루닝 비율 (0.0 ~ 1.0)
        dim: 프루닝 차원 (0: 출력, 1: 입력)
        use_l2: True면 L2 norm, False면 magnitude
    
    Returns:
        mask: 유지할 채널 (True)와 제거할 채널 (False)
    """
    if dim == 0:
        importance = torch.norm(weight, p=2, dim=tuple(range(1, len(weight.shape))))
    else:
        importance = torch.norm(weight, p=2, dim=0)
    
    num_to_prune = int(pruning_rate * len(importance))
    if num_to_prune == 0:
        return torch.ones(len(importance), dtype=torch.bool, device=weight.device)
    
    _, indices = torch.topk(importance, num_to_prune, largest=False)
    mask = torch.ones(len(importance), dtype=torch.bool, device=weight.device)
    mask[indices] = False
    
    return mask


def update_linear_layer(old_layer, mask_in=None, mask_out=None, in_size=None, out_size=None):
    """
    선형 레이어의 가중치를 마스크 또는 크기에 따라 업데이트
    
    Args:
        old_layer: 원본 Linear 레이어
        mask_in: 입력 마스크 (True: 유지, False: 제거)
        mask_out: 출력 마스크
        in_size: 새로운 입력 크기
        out_size: 새로운 출력 크기
    
    Returns:
        new_layer: 업데이트된 Linear 레이어
    """
    if mask_in is not None:
        in_features = mask_in.sum().item()
    else:
        in_features = in_size if in_size is not None else old_layer.in_features
    
    if mask_out is not None:
        out_features = mask_out.sum().item()
    else:
        out_features = out_size if out_size is not None else old_layer.out_features
    
    new_layer = nn.Linear(in_features, out_features, bias=old_layer.bias is not None)
    
    # 가중치 업데이트
    if mask_in is not None and mask_out is not None:
        new_layer.weight.data = old_layer.weight.data[mask_out, :][:, mask_in]
    elif mask_out is not None:
        new_layer.weight.data = old_layer.weight.data[mask_out, :]
    elif mask_in is not None:
        new_layer.weight.data = old_layer.weight.data[:, mask_in]
    else:
        new_layer.weight.data = old_layer.weight.data
    
    # Bias 업데이트
    if old_layer.bias is not None and mask_out is not None:
        new_layer.bias.data = old_layer.bias.data[mask_out]
    elif old_layer.bias is not None:
        new_layer.bias.data = old_layer.bias.data
    
    return new_layer


def compute_channel_importance_hessian(weight, pruning_rate, dim=1, hessian_importance=None):
    """
    Hessian 또는 L2 norm 기반 채널 중요도 계산
    
    Args:
        weight: 가중치 텐서
        pruning_rate: 프루닝 비율
        dim: 프루닝 차원
        hessian_importance: Hessian 중요도 (있으면 사용)
    
    Returns:
        mask: 유지할 채널 마스크
    """
    if hessian_importance is not None:
        # Hessian 기반: 손실에 미치는 영향도 (2차 정보) 사용
        if dim == 1:
            channel_importance = (hessian_importance * (weight ** 2)).sum(dim=0)
        else:
            channel_importance = (hessian_importance * (weight ** 2)).sum(dim=1)
    else:
        # L2 norm 기반
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
    mask[indices[:num_to_prune]] = False
    
    return mask


def compute_hessian_importance(model, layer, img_tensor, captions_batch, wm, rwm, device, num_samples=64):
    """
    Hessian 행렬을 근사하여 중요도 계산
    
    Fisher Information Matrix를 이용: F = E[g * g^T]
    여기서 g는 gradient
    """
    print(f"      🔍 Hessian 계산 중 ({num_samples}개 샘플)...")
    
    # 입력 텐서 준비
    model.eval()
    model.to(device)
    
    # Hessian 근사 (Diagonal approximation)
    hessian = None
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    for i in range(num_samples):
        idx = i % len(img_tensor)
        img = img_tensor[idx:idx+1].to(device)
        caps = captions_batch[idx:idx+1].to(device)
        
        model.zero_grad()
        
        try:
            # Forward pass
            outputs, _ = model(img, caps)
            targets = caps[:, 1:]
            outputs_trimmed = outputs[:, :targets.shape[1], :]
            
            # Loss
            vocab_size = outputs_trimmed.shape[-1]
            loss = criterion(outputs_trimmed.reshape(-1, vocab_size), targets.reshape(-1))
            
            # Backward pass
            loss.backward()
            
            # Gradient 수집
            if hasattr(layer, 'weight') and layer.weight.grad is not None:
                grad = layer.weight.grad.data
                if hessian is None:
                    hessian = grad ** 2
                else:
                    hessian += grad ** 2
        except Exception as e:
            continue
    
    if hessian is not None:
        hessian = hessian / num_samples
    
    return hessian
