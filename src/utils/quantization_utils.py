"""
양자화(Quantization) 관련 유틸리티 함수들
- Dynamic Quantization (INT8)
- Static Quantization (Calibration)
- QAT (Quantization Aware Training)
"""

import torch
import torch.nn as nn
from copy import deepcopy
from tqdm import tqdm


# ============================================================================
# 양자화 엔진 설정
# ============================================================================
def setup_quantization_engine():
    """
    PyTorch 양자화 엔진 설정
    
    PyTorch는 여러 백엔드를 지원:
    - 'fbgemm': x86 CPU (Linux/Windows)
    - 'qnnpack': ARM CPU (모바일)
    """
    try:
        # CPU에서 사용 가능한 엔진 확인
        import torch.backends.quantized as quantized_backends
        
        # fbgemm 우선 시도 (x86 CPU)
        if hasattr(torch.backends, 'quantized'):
            try:
                torch.backends.quantized.engine = 'fbgemm'
                print("✅ 양자화 엔진: fbgemm (x86 CPU)")
                return 'fbgemm'
            except:
                pass
        
        # qnnpack 시도 (ARM CPU, 폴백)
        try:
            torch.backends.quantized.engine = 'qnnpack'
            print("✅ 양자화 엔진: qnnpack (ARM CPU)")
            return 'qnnpack'
        except:
            pass
        
        # 모두 실패 시 기본값 사용
        print("⚠️ 양자화 엔진을 자동으로 선택했습니다.")
        return None
    
    except Exception as e:
        print(f"⚠️ 양자화 엔진 설정 실패: {e}")
        return None


# ============================================================================
# Dynamic Quantization
# ============================================================================
def apply_dynamic_quantization(model, dtype=torch.qint8, inplace=False):
    """
    동적 양자화 적용 (추가 학습 불필요)
    
    FP32 → INT8 자동 변환
    - CPU 추론: 2-3배 가속
    - 메모리: 4배 감소
    - 정확도: 1-2% 손실
    
    Args:
        model: 양자화할 모델
        dtype: 양자화 데이터 타입 (torch.qint8, torch.qint32)
        inplace: 원본 모델 수정 여부
    
    Returns:
        양자화된 모델
    """
    # 엔진 설정
    setup_quantization_engine()
    
    if not inplace:
        model = deepcopy(model)
    
    try:
        print(f"   🔄 동적 양자화 적용 중...")
        
        # CPU로 이동 (양자화는 CPU에서만 지원)
        model_device = next(model.parameters()).device
        model = model.cpu()
        
        # Dynamic Quantization 적용
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            qconfig_spec={torch.nn.Linear,torch.nn.Conv2d},  # Linear 레이어만 양자화
            dtype=dtype
        )
        
        # 원래 device로 복원
        quantized_model = quantized_model.to(model_device)
        
        print(f"   ✅ 동적 양자화 완료")
        return quantized_model
    
    except RuntimeError as e:
        if "NoQEngine" in str(e):
            print(f"   ❌ 양자화 엔진 오류: {e}")
            print(f"      해결: torch 재설치 또는 다른 양자화 방식 사용")
            return model
        else:
            raise


# ============================================================================
# Static Quantization
# ============================================================================
def apply_static_quantization(model, calibration_dataloader, device='cpu', inplace=False):
    """
    정적 양자화 적용 (Calibration 필요)
    
    동적 양자화보다 정확도 우수
    - CPU 추론: 3-4배 가속
    - 메모리: 4배 감소
    - 정확도: 0.5-1% 손실
    
    Args:
        model: 양자화할 모델
        calibration_dataloader: Calibration용 데이터로더
        device: 실행 device
        inplace: 원본 모델 수정 여부
    
    Returns:
        양자화된 모델
    """
    setup_quantization_engine()
    
    if not inplace:
        model = deepcopy(model)
    
    try:
        print(f"   🔄 정적 양자화 준비 중...")
        
        # CPU로 이동
        model_device = model.device if hasattr(model, 'device') else device
        model = model.cpu()
        
        # Quantization config 설정
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # Prepare (양자화 준비)
        torch.quantization.prepare(model, inplace=True)
        
        # Calibration (범위 측정)
        print(f"   📊 Calibration 진행 중...")
        model.eval()
        with torch.no_grad():
            for batch_idx, (imgs, caps) in enumerate(calibration_dataloader):
                imgs = imgs.cpu()
                try:
                    _ = model(imgs, caps)
                except:
                    # caps 없이 시도
                    _ = model(imgs)
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"      Calibration: {batch_idx + 1} batches")
                
                # 처음 50개 배치만 사용 (충분한 범위 측정)
                if batch_idx >= 50:
                    break
        
        # Convert (양자화 적용)
        print(f"   ✅ 정적 양자화 완료 (Calibration)")
        torch.quantization.convert(model, inplace=True)
        
        # 원래 device로 복원
        model = model.to(model_device)
        return model
    
    except Exception as e:
        print(f"   ❌ 정적 양자화 실패: {e}")
        return model


# ============================================================================
# QAT (Quantization Aware Training)
# ============================================================================
def apply_qat(model, train_dataloader, epochs=3, device='cpu', 
             learning_rate=1e-4, inplace=False):
    """
    양자화 인식 학습 (QAT) - 재학습으로 정확도 최대화
    
    양자화를 고려하여 모델 재학습
    - CPU 추론: 3-4배 가속
    - 메모리: 4배 감소
    - 정확도: 거의 무손실 (~0.1%)
    
    Args:
        model: 양자화할 모델
        train_dataloader: 학습용 데이터로더
        epochs: QAT 에포크 수 (보통 3-5)
        device: 실행 device
        learning_rate: 학습률 (보통 원래의 1/10)
        inplace: 원본 모델 수정 여부
    
    Returns:
        양자화된 모델
    """
    setup_quantization_engine()
    
    if not inplace:
        model = deepcopy(model)
    
    try:
        print(f"   🔄 QAT 준비 중...")
        
        # CPU로 이동
        model_device = next(model.parameters()).device
        model = model.cpu()

        # 모델을 학습 모드로
        model.train()

        # QAT config 설정
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        
        # Prepare for QAT
        torch.quantization.prepare_qat(model, inplace=True)
        
        
        # Optimizer 설정 (큰 learning rate 불필요)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate
        )
        criterion = torch.nn.CrossEntropyLoss()
        
        # QAT 학습 루프 (짧게, 3-5 에포크)
        print(f"   📚 QAT 학습 시작 ({epochs} 에포크)...")
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for batch_idx, (imgs, caps) in enumerate(tqdm(train_dataloader, 
                                                          desc=f"QAT Epoch {epoch+1}/{epochs}",
                                                          disable=True)):
                imgs = imgs.cpu()
                caps = caps.cpu()
                
                optimizer.zero_grad()
                
                try:
                    outputs, _ = model(imgs, caps)
                    targets = caps[:, 1:]
                    outputs = outputs[:, :targets.shape[1], :]
                    vocab_size = outputs.shape[-1]
                    
                    loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                except Exception as e:
                    continue
                
                # 처음 20개 배치만 사용 (충분한 학습)
                if batch_idx >= 20:
                    break
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f"      Epoch {epoch+1} Loss: {avg_loss:.4f}")
        
        # Convert (양자화 적용)
        print(f"   ✅ QAT 완료 (Convert)")
        torch.quantization.convert(model, inplace=True)
        
        # 원래 device로 복원
        model = model.to(model_device)
        return model
    
    except Exception as e:
        print(f"   ❌ QAT 실패: {e}")
        return model


# ============================================================================
# 모델 크기 비교
# ============================================================================
def get_quantized_model_size_mb(model):
    """양자화된 모델 크기 계산 (MB)"""
    param_size = 0
    
    for param in model.parameters():
        # 양자화된 parameter 확인
        if hasattr(param, 'q_scale'):  # Quantized parameter
            # INT8: 1 바이트 + scale/zero_point
            param_size += param.numel() * 1  # INT8 = 1 byte
        else:
            # 일반 parameter (FP32)
            param_size += param.numel() * 4  # FP32 = 4 bytes
    
    for buffer in model.buffers():
        if buffer.dtype in [torch.qint8, torch.uint8]:
            param_size += buffer.numel() * 1
        else:
            param_size += buffer.numel() * 4
    
    return param_size / (1024 * 1024)


def print_quantization_stats(original_model, quantized_model):
    """양자화 전후 모델 통계 출력"""
    from .pruning_utils import count_nonzero_parameters
    from .model_utils import count_parameters
    
    orig_params, _ = count_parameters(original_model)
    quant_params, _ = count_parameters(quantized_model)
    
    # 크기 추정
    orig_size = (orig_params * 4) / (1024 * 1024)  # FP32
    quant_size = (quant_params * 1) / (1024 * 1024)  # INT8 (대략)
    
    print(f"\n📊 양자화 통계:")
    print(f"   원본 모델:")
    print(f"      • 파라미터: {orig_params:,}")
    print(f"      • 크기: {orig_size:.2f} MB (FP32)")
    print(f"   양자화 모델:")
    print(f"      • 파라미터: {quant_params:,}")
    print(f"      • 크기: {quant_size:.2f} MB (INT8)")
    print(f"      • 감소율: {(1 - quant_size/orig_size)*100:.1f}%")
