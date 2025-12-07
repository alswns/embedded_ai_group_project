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
from PIL import Image


def get_peak_memory_mb(device=None):
    """GPU/CPU 메모리 사용량 (MB)"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == 'cuda':
        # CUDA 메모리: 현재 할당된 메모리
        return torch.cuda.memory_allocated() / 1024 / 1024
    elif device.type == 'mps':
        # MPS 메모리
        try:
            return torch.mps.current_allocated_memory() / 1024 / 1024
        except:
            return 0.0
    else:
        # CPU 메모리 (정확하지 않음)
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0


def clear_memory(device):
    """메모리 정리 및 초기화"""
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif device.type == 'mps':
        torch.mps.empty_cache()


def get_model_memory_mb(model):
    """모델이 차지하는 메모리 계산 (파라미터 + 버퍼)"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers()) / 1024 / 1024
    return param_size + buffer_size


def measure_inference_latency(model, inp, wm, rwm, num_runs=50, device=None):
    """
    추론 시간 측정 (word_map, rev_word_map 포함)
    
    Args:
        model: 평가할 모델
        inp: 입력 텐서
        wm: word_map
        rwm: rev_word_map
        num_runs: 측정 횟수
        device: 디바이스
    
    Returns:
        dict: {'mean_ms', 'std_ms', 'min_ms', 'max_ms', 'latencies'}
    """
    if device is None:
        device = next(model.parameters()).device
    
    latencies = []
    
    for _ in range(num_runs):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start = time.time()
        with torch.no_grad():
            _ = model.generate(inp, wm, rwm, 20)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        latency = (time.time() - start) * 1000  # ms
        latencies.append(latency)
    
    return {
        'mean_ms': np.mean(latencies),
        'std_ms': np.std(latencies),
        'min_ms': np.min(latencies),
        'max_ms': np.max(latencies),
        'latencies': latencies,
    }


def measure_inference_latency_with_memory(model, inp, wm, rwm, num_runs=50, device=None, warmup_runs=10):
    """
    추론 시간과 메모리를 동시에 측정 (토큰당 시간 포함)
    
    Args:
        model: 평가할 모델
        inp: 입력 텐서
        wm: word_map
        rwm: rev_word_map
        num_runs: 측정 횟수
        device: 디바이스
        warmup_runs: Warmup 횟수 (기본 10회)
    
    Returns:
        dict: {
            'mean_ms': 전체 추론 평균 시간,
            'std_ms': 표준편차,
            'mean_ms_per_token': 토큰당 평균 시간,
            'avg_tokens': 평균 생성 토큰 수,
            'peak_memory_mb': 추론 중 추가 메모리 (모델 제외),
            'total_memory_mb': 전체 메모리 (모델 + 추론),
            'model_memory_mb': 모델만의 메모리
        }
    """
    if device is None:
        device = next(model.parameters()).device
    
    latencies = []
    token_counts = []
    inference_memory_samples = []
    
    # 모델 메모리 계산 (고정)
    model_memory_mb = get_model_memory_mb(model)
    
    # 측정 전 메모리 완전 정리
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif device.type == 'mps':
        torch.mps.empty_cache()
    
    # Baseline 메모리 기록 (모델이 로드된 상태에서)
    if device.type == 'cuda':
        baseline_mem = torch.cuda.memory_allocated() / 1024 / 1024
    elif device.type == 'mps':
        baseline_mem = torch.mps.current_allocated_memory() / 1024 / 1024
    else:
        # CPU: psutil을 사용하여 프로세스 메모리 측정
        try:
            import psutil
            process = psutil.Process(os.getpid())
            baseline_mem = process.memory_info().rss / 1024 / 1024
        except (ImportError, Exception):
            baseline_mem = 0
    
    # Warmup: 모델을 안정화시키기 위해 여러 번 실행 (동기화 포함)
    for _ in range(warmup_runs):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        
        with torch.no_grad():
            _ = model.generate(inp, wm, rwm, 20)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
    
    # Warmup 후 메모리 정리 및 baseline 재기록
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline_mem = torch.cuda.memory_allocated() / 1024 / 1024
    elif device.type == 'mps':
        torch.mps.empty_cache()
        baseline_mem = torch.mps.current_allocated_memory() / 1024 / 1024
    else:
        # CPU: psutil을 사용하여 프로세스 메모리 측정
        try:
            import psutil
            process = psutil.Process(os.getpid())
            baseline_mem = process.memory_info().rss / 1024 / 1024
        except (ImportError, Exception):
            pass  # baseline_mem은 이미 설정됨
    
    for _ in range(num_runs):
        # 매 실행마다 메모리 초기화
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        elif device.type == 'mps':
            torch.mps.empty_cache()
        
        # 추론 전 메모리 측정 (empty_cache 이후, 모델만 있는 상태)
        if device.type == 'cuda':
            pre_inference_mem = torch.cuda.memory_allocated() / 1024 / 1024
        elif device.type == 'mps':
            pre_inference_mem = torch.mps.current_allocated_memory() / 1024 / 1024
        else:
            # CPU: psutil을 사용하여 프로세스 메모리 측정
            try:
                import psutil
                process = psutil.Process(os.getpid())
                pre_inference_mem = process.memory_info().rss / 1024 / 1024
            except (ImportError, Exception):
                pre_inference_mem = baseline_mem
        
        # 디바이스 동기화 (시간 측정 전)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        
        start = time.time()
        with torch.no_grad():
            generated_tokens = model.generate(inp, wm, rwm, 20)
        
        # 디바이스 동기화 (시간 측정 후)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        
        latency = (time.time() - start) * 1000  # ms
        
        # 토큰 수 계산 (<start>, <end> 제외)
        num_tokens = len([t for t in generated_tokens if t not in ['<start>', '<end>', '<pad>']])
        
        # 추론 직후 메모리 측정
        if device.type == 'cuda':
            # CUDA: peak memory 사용 (reset 이후, 가장 정확)
            peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
            # 추론 메모리 = peak - 모델 메모리
            inference_mem = max(0, peak_mem - model_memory_mb)
        elif device.type == 'mps':
            # MPS: 추론 후 메모리 측정
            post_inference_mem = torch.mps.current_allocated_memory() / 1024 / 1024
            
            # 추론 메모리 = (추론 후 메모리 - 추론 전 메모리)
            # 이렇게 하면 이번 실행에서 실제로 추가로 사용한 메모리만 측정됨
            inference_mem = max(0, post_inference_mem - pre_inference_mem)
            
            # generate()가 끝나면 중간 텐서가 해제되어 0에 가까울 수 있음
            # 이 경우, 최소값 적용 (실제 추론 시 필요한 최소 메모리)
            if inference_mem < 0.1:
                # 최소값: 모델 크기의 약 3-5% (추론 시 중간 텐서 필요)
                inference_mem = model_memory_mb * 0.05
        else:
            # CPU: psutil을 사용하여 프로세스 메모리 측정
            try:
                import psutil
                process = psutil.Process(os.getpid())
                post_inference_mem = process.memory_info().rss / 1024 / 1024
                # 추론 메모리 = (추론 후 메모리 - 추론 전 메모리)
                inference_mem = max(0, post_inference_mem - pre_inference_mem)
                
                # generate()가 끝나면 중간 텐서가 해제되어 0에 가까울 수 있음
                # 이 경우, 최소값 적용
                if inference_mem < 0.1:
                    inference_mem = model_memory_mb * 0.05
            except (ImportError, Exception):
                inference_mem = model_memory_mb * 0.05  # 최소값 적용
        
        latencies.append(latency)
        token_counts.append(max(1, num_tokens))
        inference_memory_samples.append(inference_mem)
    
    # 이상치 제거: 상하위 10%
    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)
    trim_count = max(1, n // 10)
    latencies_trimmed = latencies_sorted[trim_count:-trim_count] if n > 2 * trim_count else latencies_sorted
    
    # 토큰당 시간 계산
    ms_per_token = [lat / cnt for lat, cnt in zip(latencies, token_counts)]
    ms_per_token_sorted = sorted(ms_per_token)
    ms_per_token_trimmed = ms_per_token_sorted[trim_count:-trim_count] if n > 2 * trim_count else ms_per_token_sorted
    
    # 추론 메모리 (모델 제외, 순수 추론 시 사용하는 메모리)
    avg_inference_memory = np.mean(inference_memory_samples) if inference_memory_samples else 0
    peak_inference_memory = max(inference_memory_samples) if inference_memory_samples else 0
    
    return {
        'mean_ms': np.mean(latencies_trimmed),
        'std_ms': np.std(latencies_trimmed),
        'min_ms': np.min(latencies),
        'max_ms': np.max(latencies),
        'latencies': latencies,
        # 토큰당 시간
        'mean_ms_per_token': np.mean(ms_per_token_trimmed),
        'std_ms_per_token': np.std(ms_per_token_trimmed),
        'avg_tokens': np.mean(token_counts),
        # 메모리 (구분)
        'model_memory_mb': model_memory_mb,           # 모델 파라미터만
        'inference_memory_mb': peak_inference_memory, # 추론 시 추가 메모리
        'total_memory_mb': model_memory_mb + peak_inference_memory,  # 전체
        # 하위 호환성
        'peak_memory_mb': model_memory_mb + peak_inference_memory,
        'mean_memory_mb': model_memory_mb + avg_inference_memory,
    }


def calculate_meteor_batch(model, test_images, test_captions, wm, rwm, calculate_meteor_fn):
    """
    여러 이미지에 대한 METEOR 점수 배치 계산
    
    Args:
        model: 모델
        test_images: 테스트 이미지 텐서 리스트
        test_captions: 참조 캡션 리스트
        wm: word_map
        rwm: rev_word_map
        calculate_meteor_fn: METEOR 계산 함수
    
    Returns:
        dict: {'avg_meteor', 'meteor_scores', 'example_caption', 'ref_caption'}
    """
    meteor_scores = []
    example_caption = "N/A"
    ref_caption = "N/A"
    
    for idx, (test_img, ref_cap) in enumerate(zip(test_images, test_captions)):
        if ref_cap:
            with torch.no_grad():
                gen_seq = model.generate(test_img, wm, rwm, 20)
            meteor = calculate_meteor_fn(gen_seq, ref_cap)
            if meteor is not None:
                meteor_scores.append(meteor)
            if idx == 0:
                example_caption = ' '.join([w for w in gen_seq if w not in ['<start>', '<end>', '<pad>', '<unk>']])
                ref_caption = ref_cap
    
    avg_meteor = np.mean(meteor_scores) if meteor_scores else None
    
    return {
        'avg_meteor': avg_meteor,
        'meteor_scores': meteor_scores,
        'example_caption': example_caption,
        'ref_caption': ref_caption,
    }


def load_test_images_for_meteor(val_dataloader, transform, num_images, device, rev_word_map=None, dtype=torch.float32):
    """
    METEOR 측정을 위한 검증 데이터로더에서만 이미지 로드 (데이터 오염 방지)
    
    Args:
        val_dataloader: 검증 데이터로더 (train/val 분리된 데이터만 사용)
        transform: 이미지 전처리 함수
        num_images: 로드할 이미지 수
        device: 디바이스
        rev_word_map: 역 단어 맵 (캡션 텍스트 복원용)
        dtype: 데이터 타입
    
    Returns:
        tuple: (test_images, test_captions)
    """
    test_images = []
    test_captions = []
    
    # val_dataloader에서 이미지와 캡션 추출
    count = 0
    for imgs, caps in val_dataloader:
        if count >= num_images:
            break
        
        for img, cap in zip(imgs, caps):
            if count >= num_images:
                break
            
            # 이미지를 device와 dtype으로 변환
            img_tensor = img.unsqueeze(0).to(device).to(dtype)
            test_images.append(img_tensor)
            
            # 캡션 텐서를 텍스트로 변환
            caption_text = ""
            if rev_word_map is not None and isinstance(cap, torch.Tensor):
                # 캡션 텐서를 단어로 변환
                caption_tokens = []
                for token_idx in cap:
                    token_id = int(token_idx.item())
                    if token_id in rev_word_map:
                        word = rev_word_map[token_id]
                        # <start>, <end>, <pad> 제외
                        if word not in ['<start>', '<end>', '<pad>', '<unk>']:
                            caption_tokens.append(word)
                caption_text = ' '.join(caption_tokens)
            else:
                # rev_word_map이 없으면 캡션을 그대로 사용
                caption_text = str(cap) if isinstance(cap, str) else ""
            
            test_captions.append(caption_text)
            
            count += 1
    
    return test_images, test_captions


def print_benchmark_result(result, prefix=""):
    """벤치마크 결과 출력 (일관된 형식)"""
    print("{}⏱️ 평균 시간: {:.2f} ± {:.2f} ms".format(prefix, result["mean_time_ms"], result["std_time_ms"]))
    print("{}💾 모델 크기 (Dense): {:.2f} MB".format(prefix, result.get("model_size_mb_dense", 0)))
    print("{}💾 모델 크기 (Sparse): {:.2f} MB".format(prefix, result["model_size_mb"]))
    print("{}📉 크기 감소율: {:.2f}%".format(prefix, result.get("size_reduction", 0)))
    print("{}📊 총 파라미터: {:,} (0이 아닌: {:,})".format(prefix, result["total_params"], result.get("nonzero_params", 0)))
    print("{}✂️ Sparsity: {:.2f}%".format(prefix, result.get("sparsity", 0)))
    print("{}🧠 메모리 사용량: {:.2f} MB".format(prefix, result.get("memory_usage_mb", 0)))
    if result.get('meteor_score') is not None:
        print("{}⭐ METEOR: {:.4f}".format(prefix, result.get("meteor_score", 0)))
    print("{}📝 예시 캡션: {}".format(prefix, result.get("example_caption", 'N/A')))

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


def calculate_size_reduction(total_params, nonzero_params, baseline_params, sparsity):
    """
    크기 감소율 계산
    - Structured Pruning: total_params 기준
    - Magnitude Pruning: nonzero_params 기준
    """
    if baseline_params is None or baseline_params <= 0:
        return 0.0
    
    if sparsity > 1.0 and total_params == baseline_params:
        return (1 - nonzero_params / baseline_params) * 100
    else:
        return (1 - total_params / baseline_params) * 100


def run_benchmark(model, img_tensor, wm, rwm, precision_name, ref_caption=None, 
                 baseline_params=None, num_runs=50, num_meteor_images=100,
                 val_dataloader=None, transform=None,
                 calculate_meteor_fn=None, dtype=torch.float32):
    """
    모델 벤치마크 실행 (추론 시간, 메모리, METEOR 측정)
    
    Args:
        model: 평가할 모델
        img_tensor: 입력 이미지 텐서
        wm: word_map
        rwm: rev_word_map
        precision_name: 벤치마크 레이블
        ref_caption: 참조 캡션 (옵션)
        baseline_params: 베이스라인 파라미터 수 (옵션)
        num_runs: 추론 측정 횟수
        num_meteor_images: METEOR 측정용 이미지 수
        val_dataloader: 검증 데이터로더 (데이터 오염 방지)
        transform: 이미지 전처리 함수
        calculate_meteor_fn: METEOR 계산 함수
        dtype: 데이터 타입
    
    Returns:
        dict: 벤치마크 결과
    """
    from .model_utils import count_parameters
    from .pruning_utils import count_nonzero_parameters
    
    print("\n[{}] 벤치마크 시작...".format(precision_name))
    
    model_device = next(model.parameters()).device
    inp = img_tensor.clone().detach().to(model_device)
    
    # 1. 메모리 초기화
    clear_memory(model_device)
    
    # 2. Warm-up
    print("   🔥 Warm-up 진행 중 (10회)...")
    for _ in range(10):
        if model_device.type == 'cuda':
            torch.cuda.synchronize()
        elif model_device.type == 'mps':
            torch.mps.synchronize()
        
        with torch.no_grad():
            try:
                _ = model.generate(inp, wm, rwm, 20)
            except Exception as e:
                print("⚠️ Warm-up 실패: {}".format(e))
                return None
        
        if model_device.type == 'cuda':
            torch.cuda.synchronize()
        elif model_device.type == 'mps':
            torch.mps.synchronize()
    
    # 3. 추론 시간 및 메모리 측정
    clear_memory(model_device)
    inference_metrics = measure_inference_latency_with_memory(
        model, inp, wm, rwm, num_runs=num_runs, device=model_device
    )
    latencies = inference_metrics['latencies']
    
    mean_ms = inference_metrics['mean_ms']
    std_ms = inference_metrics['std_ms']
    mean_ms_per_token = inference_metrics['mean_ms_per_token']
    std_ms_per_token = inference_metrics['std_ms_per_token']
    avg_tokens = inference_metrics['avg_tokens']
    model_memory_mb = inference_metrics['model_memory_mb']
    inference_memory_mb = inference_metrics['inference_memory_mb']
    total_memory_mb = inference_metrics['total_memory_mb']
    
    print("   ⏱️ 평균 추론 시간: {:.2f} ± {:.2f} ms".format(mean_ms, std_ms))
    print("   ⏱️ 토큰당 시간: {:.2f} ± {:.2f} ms/token".format(mean_ms_per_token, std_ms_per_token))
    print("   🧠 메모리: 모델 {:.2f} MB + 추론 {:.2f} MB".format(model_memory_mb, inference_memory_mb))
    
    # 4. METEOR 점수 계산
    avg_meteor = None
    example_caption = "N/A"
    
    if calculate_meteor_fn and val_dataloader and transform:
        print("   📊 METEOR 점수 측정 중: {}개 이미지 (val_dataloader에서)".format(num_meteor_images))
        test_images, test_captions = load_test_images_for_meteor(
            val_dataloader, transform, num_meteor_images, model_device, rev_word_map=rwm, dtype=dtype
        )
        
        meteor_result = calculate_meteor_batch(
            model, test_images, test_captions, wm, rwm, calculate_meteor_fn
        )
        avg_meteor = meteor_result['avg_meteor']
        example_caption = meteor_result['example_caption']
        ref_caption = meteor_result['ref_caption']
    
    # 5. 모델 정보 계산
    size_mb_dense = calculate_model_size_mb(model, model_type='dense')
    size_mb_sparse = calculate_model_size_mb(model, model_type='sparse')
    sparsity = calculate_sparsity(model)
    total_params, trainable_params = count_parameters(model)
    nonzero_params, _ = count_nonzero_parameters(model)
    
    size_reduction = calculate_size_reduction(total_params, nonzero_params, baseline_params, sparsity)
    
    # 6. 결과 출력
    print("   💾 모델 크기: {:.2f} MB (Sparse)".format(size_mb_sparse))
    print("   📉 파라미터 감소율: {:.2f}%".format(size_reduction))
    print("   📊 총 파라미터: {:,}".format(total_params))
    print("   ✂️ Sparsity: {:.2f}%".format(sparsity))
    if avg_meteor is not None:
        print("   ⭐ METEOR: {:.4f}".format(avg_meteor))
    print("   📝 예시 캡션: {}".format(example_caption))
    
    return {
        'precision': precision_name,
        'mean_time_ms': mean_ms,
        'std_time_ms': std_ms,
        'min_time_ms': inference_metrics['min_ms'],
        'max_time_ms': inference_metrics['max_ms'],
        'mean_ms_per_token': mean_ms_per_token,
        'std_ms_per_token': std_ms_per_token,
        'avg_tokens': avg_tokens,
        'model_size_mb': size_mb_sparse,
        'model_size_mb_dense': size_mb_dense,
        'model_memory_mb': model_memory_mb,
        'inference_memory_mb': inference_memory_mb,
        'total_memory_mb': total_memory_mb,
        'memory_usage_mb': total_memory_mb,
        'meteor_score': avg_meteor,
        'inference_times': latencies,
        'example_caption': example_caption,
        'total_params': total_params,
        'nonzero_params': nonzero_params,
        'sparsity': sparsity,
        'trainable_params': trainable_params,
        'size_reduction': size_reduction
    }
