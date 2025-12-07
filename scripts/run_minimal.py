#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jetson Nano용 최소화된 이미지 캡셔닝 시스템
최대 안정성 우선
"""

import cv2
import torch
import numpy as np
import os
import threading
import tempfile
import time
import psutil
import gc
import sys

print("📦 모듈 로드 중...", file=sys.stderr)

try:
    from PIL import Image
    from torchvision import transforms
    from gtts import gTTS
    import pygame
    from src.muti_modal_model.model import MobileNetCaptioningModel
    from src.utils.quantization_utils import apply_dynamic_quantization
    print("✅ 모든 모듈 로드 완료", file=sys.stderr)
except ImportError as e:
    print("❌ 모듈 로드 실패: {}".format(e), file=sys.stderr)
    sys.exit(1)

# ============================================================================
# 환경 설정
# ============================================================================
print("⚙️  환경 설정 중...", file=sys.stderr)

# GPU 완전 비활성화
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

# CPU 최적화
torch.set_num_threads(2)
torch.set_num_interop_threads(1)

# 강제 CPU 디바이스
device = torch.device("cpu")
print("📍 디바이스: CPU", file=sys.stderr)

# 모델 경로
MODELS = {
    '1': {
        'name': 'Original Model',
        'path': 'models/lightweight_captioning_model.pth',
        'fallback': 'lightweight_captioning_model.pth'
    },
    '2': {
        'name': 'Pruned Model',
        'path': 'pruning_results/Pruning_epoch_1_checkpoint.pt',
        'fallback': None
    }
}

QUANTIZE_OPTIONS = {
    '1': {'name': 'FP32 (원본)', 'enabled': False},
    '2': {'name': 'FP16', 'enabled': True},
    '3': {'name': 'INT8', 'enabled': True}
}

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

print("✅ 환경 설정 완료", file=sys.stderr)

# ============================================================================
# 유틸리티 함수
# ============================================================================

def select_model():
    """모델 선택"""
    print("\n모델 선택:")
    for key, info in MODELS.items():
        exists = "✅" if os.path.exists(info['path']) else "❌"
        print("  {}. {} {}".format(key, info['name'], exists))
    
    while True:
        choice = input("선택 (1-2): ").strip()
        if choice in MODELS:
            return choice
        print("잘못된 입력입니다.")

def select_quantization():
    """양자화 옵션 선택"""
    print("\n양자화 옵션:")
    for key, info in QUANTIZE_OPTIONS.items():
        status = "✅" if info['enabled'] else "❌"
        print("  {}. {} {}".format(key, info['name'], status))
    
    while True:
        choice = input("선택 (1-3): ").strip()
        if choice in QUANTIZE_OPTIONS:
            return choice
        print("잘못된 입력입니다.")

def load_model(model_choice):
    """모델 로드"""
    info = MODELS[model_choice]
    path = info['path']
    
    if not os.path.exists(path):
        if info['fallback'] and os.path.exists(info['fallback']):
            path = info['fallback']
        else:
            print("❌ 모델을 찾을 수 없습니다: {}".format(info['path']))
            return None, None, None, None
    
    try:
        print("📂 모델 로드: {}".format(path))
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        
        if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
            print("❌ 유효하지 않은 모델 파일")
            return None, None, None, None
        
        word_map = checkpoint.get('word_map')
        rev_word_map = checkpoint.get('rev_word_map')
        vocab_size = checkpoint.get('vocab_size')
        
        if not (word_map and rev_word_map and vocab_size):
            print("❌ 단어장 정보 없음")
            return None, None, None, None
        
        # 모델 생성
        state_dict = checkpoint['model_state_dict']
        decoder_dim = checkpoint.get('decoder_dim', 512)
        attention_dim = checkpoint.get('attention_dim', 256)
        
        try:
            model = MobileNetCaptioningModel(
                vocab_size=vocab_size,
                embed_dim=300,
                decoder_dim=decoder_dim,
                attention_dim=attention_dim
            )
            model = model.to(device)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
        except Exception as e:
            print("❌ 모델 생성 실패: {}".format(e))
            return None, None, None, None
        
        # 메모리 정리
        del checkpoint, state_dict
        gc.collect()
        
        print("✅ 모델 로드 완료")
        return model, word_map, rev_word_map, info['name']
        
    except Exception as e:
        print("❌ 모델 로드 오류: {}".format(e))
        import traceback
        traceback.print_exc()
        return None, None, None, None

def apply_quantization(model, choice, name):
    """양자화 적용"""
    if choice == '1':
        print("FP32 (양자화 없음)")
        return model.cpu(), name
    elif choice == '2':
        print("FP16은 CPU에서 미지원 - FP32 사용")
        return model.cpu(), name
    elif choice == '3':
        print("INT8 적용 시도...")
        try:
            model = apply_dynamic_quantization(model)
            return model.cpu(), name + " + INT8"
        except Exception as e:
            print("INT8 실패 - FP32 사용: {}".format(e))
            return model.cpu(), name
    
    return model.cpu(), name

def generate_caption(model, word_map, rev_word_map, frame):
    """캡션 생성"""
    image_tensor = None
    try:
        model = model.cpu()
        model.eval()
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        image_tensor = transform(pil_image).unsqueeze(0)
        
        start_time = time.time()
        with torch.no_grad():
            try:
                generated_words = model.generate(image_tensor, word_map, rev_word_map, max_len=50)
            except Exception as e:
                print("⚠️  추론 오류: {}".format(e))
                gc.collect()
                return None, 0.0
        
        inference_time = (time.time() - start_time) * 1000
        caption = ' '.join([w for w in generated_words 
                           if w not in ['<start>', '<end>', '<pad>', '<unk>']])
        
        return caption, inference_time
    except Exception as e:
        print("캡션 생성 오류: {}".format(e))
        return None, 0.0
    finally:
        if image_tensor is not None:
            del image_tensor
        gc.collect()

# ============================================================================
# 메인
# ============================================================================

def main():
    print("\n" + "="*70)
    print("📸 이미지 캡셔닝 시스템 (Jetson Nano)")
    print("="*70)
    
    # 모델 선택
    model_choice = select_model()
    
    # 모델 로드
    model, word_map, rev_word_map, model_name = load_model(model_choice)
    if model is None:
        print("❌ 모델 로드 실패")
        return
    
    # 양자화 선택
    quant_choice = select_quantization()
    model, model_name = apply_quantization(model, quant_choice, model_name)
    
    print("\n" + "="*70)
    print("모델: {}".format(model_name))
    print("="*70)
    print("\n키 입력: s (캡션), r (재생), q (종료)\n")
    
    # 카메라 시작
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 카메라 없음")
        return
    
    last_caption = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imshow('Image Captioning', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            caption, inf_time = generate_caption(model, word_map, rev_word_map, frame)
            if caption:
                last_caption = caption
                print("\n캡션: {}".format(caption))
                print("시간: {:.2f}ms\n".format(inf_time))
        elif key == ord('r') and last_caption:
            print("\n이전 캡션: {}".format(last_caption))
    
    cap.release()
    cv2.destroyAllWindows()
    print("종료")

if __name__ == "__main__":
    main()
