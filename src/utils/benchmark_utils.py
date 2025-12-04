"""
벤치마크 관련 유틸리티 함수들
- 시간 측정
- 메모리 측정
- METEOR 점수 계산
"""

import os
import gc
import time
import torch
import numpy as np
from pathlib import Path


def get_peak_memory_mb():
    """현재 메모리 사용량 (MB)"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


def calculate_model_size_mb(model, model_type='dense'):
    """
    모델 크기 계산 (MB)
    
    Args:
        model: PyTorch 모델
        model_type: 'dense' 또는 'sparse'
    
    Returns:
        float: 모델 크기 (MB)
    """
    if model_type == 'dense':
        param_size = sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024)
        buffer_size = sum(b.numel() for b in model.buffers()) * 4 / (1024 * 1024)
        return param_size + buffer_size
    else:
        # Sparse 모델: 0이 아닌 파라미터만 계산
        nonzero_count = 0
        for p in model.parameters():
            nonzero_count += (p != 0).sum().item()
        return nonzero_count * 4 / (1024 * 1024)


def calculate_sparsity(model):
    """
    모델의 희소성 계산 (%)
    
    Returns:
        float: 0인 파라미터의 비율 (%)
    """
    total_params = sum(p.numel() for p in model.parameters())
    zero_params = sum((p == 0).sum().item() for p in model.parameters())
    return (zero_params / total_params * 100) if total_params > 0 else 0.0


def measure_inference_time(model, input_data, num_runs=50, warmup=5):
    """
    추론 시간 측정
    
    Args:
        model: 평가할 모델
        input_data: 입력 데이터
        num_runs: 측정 횟수
        warmup: Warm-up 횟수
    
    Returns:
        dict: {'mean_ms': float, 'std_ms': float, 'min_ms': float, 'max_ms': float}
    """
    device = next(model.parameters()).device
    
    # Warm-up
    with torch.no_grad():
        for _ in range(warmup):
            _ = model.generate(input_data.clone().to(device), None, None, 20)
    
    # GC 한 번만 수행
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    latencies = []
    
    for _ in range(num_runs):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start = time.time()
        with torch.no_grad():
            _ = model.generate(input_data.clone().to(device), None, None, 20)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        latency = (time.time() - start) * 1000  # ms
        latencies.append(latency)
    
    return {
        'mean_ms': np.mean(latencies),
        'std_ms': np.std(latencies),
        'min_ms': np.min(latencies),
        'max_ms': np.max(latencies),
    }


def calculate_parameter_reduction(original_params, pruned_params):
    """
    파라미터 감소율 계산 (%)
    
    Args:
        original_params: 원본 파라미터 수
        pruned_params: 프루닝 후 파라미터 수
    
    Returns:
        float: 감소율 (%)
    """
    if original_params == 0:
        return 0.0
    return (1 - pruned_params / original_params) * 100


def calculate_flops_reduction(original_flops, pruned_flops):
    """
    FLOPs 감소율 계산 (%)
    
    Args:
        original_flops: 원본 FLOPs
        pruned_flops: 프루닝 후 FLOPs
    
    Returns:
        float: 감소율 (%)
    """
    if original_flops == 0:
        return 0.0
    return (1 - pruned_flops / original_flops) * 100
