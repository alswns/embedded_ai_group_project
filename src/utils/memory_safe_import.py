#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
메모리 안전 import 유틸 (최소화 버전)
프로젝트 모듈 지연 로드
"""

import sys
import gc

def check_available_memory(min_mb=800):
    """사용 가능 메모리 확인"""
    try:
        import psutil
        available = psutil.virtual_memory().available / 1024 / 1024
        print("📊 메모리: {:.0f}MB (필요: {}MB)".format(available, min_mb), file=sys.stderr)
        
        if available < min_mb:
            raise MemoryError("메모리 부족")
        return available
    except ImportError:
        print("⚠️  psutil 없음, 메모리 체크 스킵", file=sys.stderr)
        return 1000.0
    except Exception as e:
        print("⚠️  메모리 확인 실패: {}".format(e), file=sys.stderr)
        return 1000.0

def pre_cleanup():
    """메모리 정리"""
    try:
        gc.collect()
    except:
        pass

def aggressive_memory_cleanup():
    """적극적 메모리 정리 (모델 생성 전)"""
    print("🧹 적극적 메모리 정리 시작...", file=sys.stderr)
    
    try:
        # 1단계: 가비지 컬렉션 (여러 번)
        for i in range(3):
            gc.collect()
            print("  {}단계: gc.collect() 완료".format(i+1), file=sys.stderr)
        
        # 2단계: 캐시 정리
        import torch
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("  CUDA 캐시 정리 완료", file=sys.stderr)
        
        # 3단계: numpy 캐시 정리
        try:
            import numpy as np
            if hasattr(np, 'seterr'):
                np.seterr(all='ignore')
            print("  numpy 설정 완료", file=sys.stderr)
        except:
            pass
        
        # 4단계: 사용 가능 메모리 확인
        try:
            import psutil
            available = psutil.virtual_memory().available / 1024 / 1024
            print("✅ 정리 후 메모리: {:.0f}MB".format(available), file=sys.stderr)
        except:
            pass
            
    except Exception as e:
        print("⚠️  정리 중 오류: {}".format(e), file=sys.stderr)

def safe_model_instantiation(model_class, vocab_size, embed_dim, decoder_dim, attention_dim):
    """안전한 모델 인스턴스 생성"""
    print("🔧 안전한 모델 생성...", file=sys.stderr)
    
    try:
        # Step 1: 메모리 정리
        print("  1️⃣  메모리 정리...", file=sys.stderr)
        aggressive_memory_cleanup()
        
        # Step 2: 메모리 충분성 확인
        print("  2️⃣  메모리 확인...", file=sys.stderr)
        check_available_memory(min_mb=1200)
        
        # Step 3: PyTorch 메모리 설정
        print("  3️⃣  PyTorch 최적화...", file=sys.stderr)
        import torch
        torch.no_grad().__enter__()  # no_grad 모드 진입
        
        # Step 4: 모델 생성 (메모리 할당 최소화)
        print("  4️⃣  모델 인스턴스 생성...", file=sys.stderr)
        model = model_class(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            decoder_dim=decoder_dim,
            attention_dim=attention_dim
        )
        print("     ✅ 생성 완료", file=sys.stderr)
        
        # Step 5: 모델을 CPU로
        print("  5️⃣  CPU 전환...", file=sys.stderr)
        model = model.cpu()
        model.eval()
        print("     ✅ 설정 완료", file=sys.stderr)
        
        # Step 6: 메모리 정리
        print("  6️⃣  메모리 정리...", file=sys.stderr)
        gc.collect()
        
        print("✅ 모델 생성 완료", file=sys.stderr)
        return model
        
    except MemoryError as e:
        print("❌ 메모리 부족: {}".format(e), file=sys.stderr)
        raise
    except Exception as e:
        print("❌ 모델 생성 실패: {}".format(e), file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise

def lazy_load_model_class():
    """MobileNetCaptioningModel 지연 로드"""
    try:
        # 메모리 확인
        check_available_memory(min_mb=1000)
        
        # 정리
        pre_cleanup()
        
        # Import
        print("로드 중: MobileNetCaptioningModel", file=sys.stderr)
        from src.muti_modal_model.model import Model
        print("✅ 로드 완료", file=sys.stderr)
        return Model
        
    except MemoryError as e:
        print("❌ 메모리 부족", file=sys.stderr)
        raise
    except ImportError as e:
        print("❌ Import 실패: {}".format(e), file=sys.stderr)
        raise
    except Exception as e:
        print("❌ 오류: {}".format(e), file=sys.stderr)
        raise

def lazy_load_quantization():
    """apply_dynamic_quantization 지연 로드"""
    try:
        # 메모리 확인
        check_available_memory(min_mb=500)
        
        # 정리
        pre_cleanup()
        
        # Import
        print("로드 중: apply_dynamic_quantization", file=sys.stderr)
        from src.utils.quantization_utils import apply_dynamic_quantization
        print("✅ 로드 완료", file=sys.stderr)
        return apply_dynamic_quantization
        
    except MemoryError as e:
        print("❌ 메모리 부족", file=sys.stderr)
        raise
    except ImportError as e:
        print("❌ Import 실패: {}".format(e), file=sys.stderr)
        raise
    except Exception as e:
        print("❌ 오류: {}".format(e), file=sys.stderr)
        raise

# 간편 함수
def load_model_class():
    """모델 클래스 로드"""
    print('모델 클래스 로드 요청 받음', file=sys.stderr)
    return lazy_load_model_class()

def load_quantization_func():
    """양자화 함수 로드"""
    return lazy_load_quantization()


