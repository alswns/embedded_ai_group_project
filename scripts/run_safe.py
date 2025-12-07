#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jetson Nano용 안정화된 이미지 캡셔닝
torchvision 없이 수동 이미지 처리
"""

import cv2
import numpy as np
import os
import threading
import tempfile
import time
import psutil
import gc
import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print("📦 안전 모듈 로드...", file=sys.stderr)

# 1. 필수 모듈만 로드
try:
    import torch
    print("✅ torch 로드 완료", file=sys.stderr)
except ImportError as e:
    print("❌ PyTorch 필요: {}".format(e), file=sys.stderr)
    sys.exit(1)

# 2. PIL만 로드 (매우 안전)
try:
    from PIL import Image
    print("✅ PIL 로드 완료", file=sys.stderr)
except ImportError as e:
    print("❌ Pillow 필요: {}".format(e), file=sys.stderr)
    sys.exit(1)

# 3. 프로젝트 모듈 (torchvision 없이)
try:
    from src.utils.safe_model_loader import load_model_safe
    print("✅ 프로젝트 모듈 로드 완료", file=sys.stderr)
    from src.utils.quantization_utils import apply_dynamic_quantization
    print("✅ quantization_utils 로드 완료", file=sys.stderr)
    
except ImportError as e:
    print("❌ 프로젝트 모듈 오류: {}".format(e), file=sys.stderr)
    sys.exit(1)

# ============================================================================
# 환경 설정 (CRITICAL)
# ============================================================================
print("⚙️  환경 설정...", file=sys.stderr)

os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

torch.set_num_threads(2)
torch.set_num_interop_threads(1)

device = torch.device("cpu")
print("✅ 환경 설정 완료 (CPU 모드)", file=sys.stderr)

# ============================================================================
# 이미지 전처리 (torchvision 대체)
# ============================================================================

def preprocess_image(frame):
    """
    OpenCV BGR 프레임을 PyTorch 텐서로 변환
    torchvision을 사용하지 않음
    """
    # BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # PIL Image로 변환
    pil_image = Image.fromarray(rgb_frame)
    
    # 224x224로 리사이즈
    pil_image = pil_image.resize((224, 224), Image.BILINEAR)
    
    # numpy array로 변환
    image_array = np.array(pil_image, dtype=np.float32)
    
    # 정규화 (ImageNet 평균/표준편차)
    image_array = image_array / 255.0
    image_array -= np.array([0.485, 0.456, 0.406], dtype=np.float32)
    image_array /= np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    # CHW 형식으로 변환 (H, W, C) → (C, H, W)
    image_array = np.transpose(image_array, (2, 0, 1))
    
    # PyTorch 텐서로 변환
    image_tensor = torch.from_numpy(image_array).float()
    
    # 배치 차원 추가
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor

# ============================================================================
# 설정
# ============================================================================

MODELS = {
    '1': {
        'name': 'Original Model',
        'path': 'models/lightweight_captioning_model.pth',
    },
    '2': {
        'name': 'Pruned Model',
        'path': 'pruning_results/Pruning_epoch_1_checkpoint.pt',
    }
}

QUANTIZE_OPTIONS = {
    '1': {'name': 'FP32 (원본)'},
    '2': {'name': 'FP16 (미지원)'},
    '3': {'name': 'INT8'}
}

# ============================================================================
# 유틸리티
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
    """양자화 선택"""
    print("\n양자화 옵션:")
    for key, info in QUANTIZE_OPTIONS.items():
        print("  {}. {}".format(key, info['name']))
    
    while True:
        choice = input("선택 (1-3): ").strip()
        if choice in QUANTIZE_OPTIONS:
            return choice
        print("잘못된 입력입니다.")

def load_model(model_choice):
    """모델 로드 (안전한 로더 사용)"""
    info = MODELS[model_choice]
    path = info['path']
    
    if not os.path.exists(path):
        print("❌ 모델 파일 없음: {}".format(path))
        return None, None, None, None
    
    try:
        print("\n📂 모델 로드 중: {}".format(path))
        model, word_map, rev_word_map = load_model_safe(path)
        
        if model is None:
            print("❌ 모델 로드 실패")
            return None, None, None, None
        
        print("✅ 모델 로드 완료")
        return model, word_map, rev_word_map, info['name']
        
    except Exception as e:
        print("❌ 예상 불가능한 오류: {}".format(e), file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return None, None, None, None

def apply_quantization(model, choice):
    """양자화 적용"""
    try:
        if choice == '1':
            print("FP32 (양자화 없음)")
        elif choice == '2':
            print("FP16은 CPU에서 미지원 - FP32 사용")
        elif choice == '3':
            print("INT8 적용...")
            model = apply_dynamic_quantization(model)
    except Exception as e:
        print("⚠️  {}".format(e))
    
    model = model.cpu()
    model.eval()
    return model

def generate_caption(model, word_map, rev_word_map, frame):
    """캡션 생성"""
    try:
        # 이미지 전처리 (torchvision 대체)
        image_tensor = preprocess_image(frame)
        
        # 캡션 생성
        start = time.time()
        with torch.no_grad():
            generated = model.generate(image_tensor, word_map, rev_word_map, max_len=50)
        elapsed = (time.time() - start) * 1000
        
        caption = ' '.join([w for w in generated 
                           if w not in ['<start>', '<end>', '<pad>', '<unk>']])
        
        del image_tensor
        gc.collect()
        
        return caption, elapsed
    except Exception as e:
        print("⚠️  {}".format(e))
        gc.collect()
        return None, 0.0

# ============================================================================
# 메인
# ============================================================================

def main():
    print("\n" + "="*70)
    print("📸 이미지 캡셔닝 시스템")
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
    model = apply_quantization(model, quant_choice)
    
    print("\n모델: {}".format(model_name))
    print("="*70)
    print("\n키: s (캡션), r (재생), q (종료)\n")
    
    # 카메라
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 카메라 없음")
        return
    
    last_caption = None
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            cv2.imshow('Captioning', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                caption, elapsed = generate_caption(model, word_map, rev_word_map, frame)
                if caption:
                    last_caption = caption
                    print("\n📝 {}".format(caption))
                    print("⏱️  {:.1f}ms\n".format(elapsed))
            elif key == ord('r') and last_caption:
                print("\n📝 {}".format(last_caption))
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    print("종료")

if __name__ == "__main__":
    main()
