#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
메모리 안전 import 유틸
프로젝트 모듈 지연 로드 및 메모리 관리
"""

import sys
import gc
import psutil

def check_available_memory(min_mb=800):
    """사용 가능 메모리 확인"""
    try:
        available = psutil.virtual_memory().available / 1024 / 1024
        print("📊 사용 가능 메모리: {:.0f}MB (필요: {}MB)".format(available, min_mb), file=sys.stderr)
        
        if available < min_mb:
            raise MemoryError("메모리 부족: {:.0f}MB < {}MB".format(available, min_mb))
        
        return available
    except Exception as e:
        print("⚠️  메모리 확인 실패: {}".format(e), file=sys.stderr)
        return 0.0

def pre_cleanup():
    """Import 전 메모리 정리"""
    print("🧹 메모리 정리 중...", file=sys.stderr)
    try:
        gc.collect()
        print("✅ 정리 완료", file=sys.stderr)
    except Exception as e:
        print("⚠️  정리 실패: {}".format(e), file=sys.stderr)

def lazy_load_model_class():
    """MobileNetCaptioningModel 지연 로드"""
    print("📦 모델 클래스 로드 중...", file=sys.stderr)
    
    try:
        # 메모리 확인
        check_available_memory(min_mb=1000)
        
        # 정리
        pre_cleanup()
        
        # Import
        from src.muti_modal_model.model import MobileNetCaptioningModel
        print("✅ 모델 클래스 로드 완료", file=sys.stderr)
        return MobileNetCaptioningModel
        
    except MemoryError as e:
        print("❌ 메모리 부족: {}".format(e), file=sys.stderr)
        raise
    except ImportError as e:
        print("❌ Import 실패: {}".format(e), file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise
    except Exception as e:
        print("❌ 예상 불가능한 오류: {}".format(e), file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise

def lazy_load_quantization():
    """apply_dynamic_quantization 지연 로드"""
    print("📦 양자화 함수 로드 중...", file=sys.stderr)
    
    try:
        # 메모리 확인
        check_available_memory(min_mb=500)
        
        # 정리
        pre_cleanup()
        
        # Import
        from src.utils.quantization_utils import apply_dynamic_quantization
        print("✅ 양자화 함수 로드 완료", file=sys.stderr)
        return apply_dynamic_quantization
        
    except MemoryError as e:
        print("❌ 메모리 부족: {}".format(e), file=sys.stderr)
        raise
    except ImportError as e:
        print("❌ Import 실패: {}".format(e), file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise
    except Exception as e:
        print("❌ 예상 불가능한 오류: {}".format(e), file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise

class LazyModelLoader:
    """모델 지연 로드 래퍼"""
    def __init__(self):
        self._model_class = None
        self._quantization_func = None
        self.memory_checks = []
    
    def get_model_class(self):
        """모델 클래스 취득 (처음 호출 시 로드)"""
        if self._model_class is None:
            print("\n⏱️  모델 클래스 첫 로드 (지연)...", file=sys.stderr)
            try:
                check_available_memory(min_mb=1200)
                pre_cleanup()
                self._model_class = lazy_load_model_class()
            except Exception as e:
                print("❌ 모델 클래스 로드 실패: {}".format(e), file=sys.stderr)
                raise
        
        return self._model_class
    
    def get_quantization_func(self):
        """양자화 함수 취득 (처음 호출 시 로드)"""
        if self._quantization_func is None:
            print("\n⏱️  양자화 함수 첫 로드 (지연)...", file=sys.stderr)
            try:
                check_available_memory(min_mb=600)
                pre_cleanup()
                self._quantization_func = lazy_load_quantization()
            except Exception as e:
                print("❌ 양자화 함수 로드 실패: {}".format(e), file=sys.stderr)
                raise
        
        return self._quantization_func
    
    def log_memory_check(self, stage, available_mb):
        """메모리 체크 로깅"""
        self.memory_checks.append({
            'stage': stage,
            'available_mb': available_mb
        })

# 글로벌 인스턴스
_lazy_loader = None

def get_lazy_loader():
    """전역 지연 로더 취득"""
    global _lazy_loader
    if _lazy_loader is None:
        _lazy_loader = LazyModelLoader()
    return _lazy_loader

# 간편 함수
def load_model_class():
    """모델 클래스 로드 (간편 함수)"""
    return get_lazy_loader().get_model_class()

def load_quantization_func():
    """양자화 함수 로드 (간편 함수)"""
    return get_lazy_loader().get_quantization_func()

if __name__ == '__main__':
    # 테스트
    print("테스트 모드:\n", file=sys.stderr)
    
    try:
        print("1️⃣  메모리 확인...", file=sys.stderr)
        check_available_memory(min_mb=500)
        
        print("\n2️⃣  모델 클래스 로드...", file=sys.stderr)
        ModelClass = load_model_class()
        print("✅ 모델 클래스: {}".format(ModelClass.__name__), file=sys.stderr)
        
        print("\n3️⃣  양자화 함수 로드...", file=sys.stderr)
        quant_func = load_quantization_func()
        print("✅ 양자화 함수: {}".format(quant_func.__name__), file=sys.stderr)
        
        print("\n✅ 모든 테스트 통과!", file=sys.stderr)
    except Exception as e:
        print("\n❌ 테스트 실패: {}".format(e), file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
