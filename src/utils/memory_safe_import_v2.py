#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
메모리 안전 import 유틸 (초경량 버전)
프로젝트 모듈 지연 로드 - 세그멘테이션 오류 방지
"""

import sys
import gc

def pre_cleanup():
    """메모리 정리"""
    try:
        gc.collect()
    except:
        pass

def safe_model_instantiation(model_class, vocab_size, embed_dim, decoder_dim, attention_dim):
    """초안전한 모델 인스턴스 생성"""
    
    # 메모리 정리 (여러 번)
    pre_cleanup()
    pre_cleanup()
    pre_cleanup()
    
    try:
        # 모델 생성
        model = model_class(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            decoder_dim=decoder_dim,
            attention_dim=attention_dim
        )
        
        # CPU 전환
        model = model.cpu()
        model.eval()
        
        # 메모리 정리
        gc.collect()
        
        return model
        
    except Exception as e:
        print("❌ 모델 생성 실패: {}".format(e), file=sys.stderr)
        raise

def lazy_load_model_class():
    """MobileNetCaptioningModel 지연 로드"""
    try:
        pre_cleanup()
        from src.muti_modal_model.model import MobileNetCaptioningModel
        return MobileNetCaptioningModel
    except Exception as e:
        print("❌ Import 실패: {}".format(e), file=sys.stderr)
        raise

def lazy_load_quantization():
    """apply_dynamic_quantization 지연 로드"""
    try:
        pre_cleanup()
        from src.utils.quantization_utils import apply_dynamic_quantization
        return apply_dynamic_quantization
    except Exception as e:
        print("❌ Import 실패: {}".format(e), file=sys.stderr)
        raise

# 간편 함수
def load_model_class():
    """모델 클래스 로드"""
    return lazy_load_model_class()

def load_quantization_func():
    """양자화 함수 로드"""
    return lazy_load_quantization()
