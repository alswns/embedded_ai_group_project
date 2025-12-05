"""
모델 관련 유틸리티 함수
"""
import torch
import psutil
import os

def count_parameters(model):
    """모델 파라미터 개수 계산"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def get_model_size_mb(model, sparse=False):
    """모델 파라미터 + 버퍼 크기 계산 (MB)
    
    Args:
        model: 모델
        sparse: True면 실제 0이 아닌 파라미터만 계산 (Pruning 후 실제 크기)
    """
    param_size = 0
    buffer_size = 0
    
    if sparse:
        # Sparse format: 실제 0이 아닌 값만 계산
        for param in model.parameters():
            if param.is_sparse:
                # Sparse tensor인 경우
                param_size += param._values().numel() * param.element_size()
                # 인덱스도 저장해야 하므로 추가
                param_size += param._indices().numel() * param._indices().element_size()
            else:
                # Dense tensor인 경우 0이 아닌 값만 계산
                nonzero = param.nonzero()
                if nonzero.numel() > 0:
                    # 0이 아닌 값의 개수
                    nonzero_count = (param != 0).sum().item()
                    param_size += nonzero_count * param.element_size()
                    # 인덱스 저장을 위한 오버헤드 (간단한 추정)
                    param_size += nonzero_count * 4  # 인덱스 저장 오버헤드 (4 bytes per index)
    else:
        # Dense format: 모든 파라미터 계산
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    return (param_size + buffer_size) / 1024 / 1024

def count_nonzero_parameters(model):
    """0이 아닌 파라미터 개수 계산 (프루닝 후)"""
    nonzero_params = 0
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
        nonzero_params += param.nonzero().size(0) if param.numel() > 0 else 0
    return nonzero_params, total_params

