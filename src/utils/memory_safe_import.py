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

def lazy_load_model_class():
    """MobileNetCaptioningModel 지연 로드"""
    try:
        # 메모리 확인
        check_available_memory(min_mb=1000)
        
        # 정리
        pre_cleanup()
        
        # Import
        print("로드 중: MobileNetCaptioningModel", file=sys.stderr)
        from src.muti_modal_model.model import MobileNetCaptioningModel
        print("✅ 로드 완료", file=sys.stderr)
        return MobileNetCaptioningModel
        
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
    return lazy_load_model_class()

def load_quantization_func():
    """양자화 함수 로드"""
    return lazy_load_quantization()

