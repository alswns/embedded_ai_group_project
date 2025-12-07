#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jetson Nano용 최대 안정성 모델 로더
모델 로드 단계별 디버깅
"""

import torch
import os
import gc
import sys

def safe_load_checkpoint(path):
    """안전한 체크포인트 로드"""
    print("  체크포인트 로드...", file=sys.stderr)
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        print("    ✅ 로드 성공", file=sys.stderr)
        return checkpoint
    except Exception as e:
        print("    ❌ 로드 실패: {}".format(e), file=sys.stderr)
        return None

def safe_extract_metadata(checkpoint):
    """안전한 메타데이터 추출"""
    print("  메타데이터 추출...", file=sys.stderr)
    try:
        if not isinstance(checkpoint, dict):
            print("    ❌ checkpoint가 dict가 아님", file=sys.stderr)
            return None
        
        if 'model_state_dict' not in checkpoint:
            print("    ❌ model_state_dict 없음", file=sys.stderr)
            return None
        
        word_map = checkpoint.get('word_map')
        rev_word_map = checkpoint.get('rev_word_map')
        vocab_size = checkpoint.get('vocab_size')
        
        if not (word_map and rev_word_map and vocab_size):
            print("    ❌ 필수 정보 부족", file=sys.stderr)
            return None
        
        state_dict = checkpoint['model_state_dict']
        decoder_dim = checkpoint.get('decoder_dim', 512)
        attention_dim = checkpoint.get('attention_dim', 256)
        
        metadata = {
            'word_map': word_map,
            'rev_word_map': rev_word_map,
            'vocab_size': vocab_size,
            'state_dict': state_dict,
            'decoder_dim': decoder_dim,
            'attention_dim': attention_dim
        }
        
        print("    ✅ 추출 성공 (vocab={})".format(vocab_size), file=sys.stderr)
        return metadata
    except Exception as e:
        print("    ❌ 추출 실패: {}".format(e), file=sys.stderr)
        return None

def safe_create_model(vocab_size, decoder_dim, attention_dim):
    """안전한 모델 생성"""
    print("  모델 생성...", file=sys.stderr)
    try:
        from src.muti_modal_model.model import MobileNetCaptioningModel
        
        model = MobileNetCaptioningModel(
            vocab_size=vocab_size,
            embed_dim=300,
            decoder_dim=decoder_dim,
            attention_dim=attention_dim
        )
        
        print("    ✅ 생성 성공", file=sys.stderr)
        return model
    except Exception as e:
        print("    ❌ 생성 실패: {}".format(e), file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return None

def safe_load_state_dict(model, state_dict):
    """안전한 가중치 로드"""
    print("  가중치 로드...", file=sys.stderr)
    try:
        model.load_state_dict(state_dict, strict=False)
        print("    ✅ 로드 성공", file=sys.stderr)
        return True
    except Exception as e:
        print("    ⚠️  로드 부분 실패: {} (계속 진행)".format(e), file=sys.stderr)
        return False

def safe_setup_eval(model):
    """안전한 평가 모드 설정"""
    print("  평가 모드 설정...", file=sys.stderr)
    try:
        model = model.cpu()
        model.eval()
        print("    ✅ 설정 성공", file=sys.stderr)
        return model
    except Exception as e:
        print("    ❌ 설정 실패: {}".format(e), file=sys.stderr)
        return None

def safe_cleanup(checkpoint=None, state_dict=None):
    """안전한 메모리 정리"""
    print("  메모리 정리...", file=sys.stderr)
    try:
        if checkpoint is not None:
            del checkpoint
        if state_dict is not None:
            del state_dict
        gc.collect()
        print("    ✅ 정리 성공", file=sys.stderr)
    except:
        pass

def load_model_safe(path):
    """완벽하게 안전한 모델 로드"""
    print("\n📂 모델 로드 시작: {}".format(path), file=sys.stderr)
    
    if not os.path.exists(path):
        print("❌ 파일 없음: {}".format(path))
        return None, None, None
    
    # Step 1: 체크포인트 로드
    checkpoint = safe_load_checkpoint(path)
    if checkpoint is None:
        return None, None, None
    
    # Step 2: 메타데이터 추출
    metadata = safe_extract_metadata(checkpoint)
    if metadata is None:
        safe_cleanup(checkpoint)
        return None, None, None
    
    # Step 3: 모델 생성
    model = safe_create_model(
        metadata['vocab_size'],
        metadata['decoder_dim'],
        metadata['attention_dim']
    )
    if model is None:
        safe_cleanup(checkpoint, metadata['state_dict'])
        return None, None, None
    
    # Step 4: 가중치 로드
    safe_load_state_dict(model, metadata['state_dict'])
    
    # Step 5: 평가 모드
    model = safe_setup_eval(model)
    if model is None:
        safe_cleanup(checkpoint, metadata['state_dict'])
        return None, None, None
    
    # Step 6: 메모리 정리
    safe_cleanup(checkpoint, metadata['state_dict'])
    
    print("✅ 모델 로드 완료")
    return model, metadata['word_map'], metadata['rev_word_map']

if __name__ == '__main__':
    # 테스트
    print("\n테스트 모드:\n")
    
    model_path = 'models/lightweight_captioning_model.pth'
    if os.path.exists(model_path):
        model, word_map, rev_word_map = load_model_safe(model_path)
        if model:
            print("\n✅ 모델 로드 성공!")
        else:
            print("\n❌ 모델 로드 실패")
    else:
        print("모델 파일 없음: {}".format(model_path))
