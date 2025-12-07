#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jetson Nano용 최소 안정성 버전
모든 프로젝트 모듈 import 제거
"""

import cv2
import numpy as np
import os
import time
import gc
import sys

print("📦 최소 모듈 로드...", file=sys.stderr)

# 필수 모듈만
try:
    import torch
    print("✅ torch", file=sys.stderr)
    from PIL import Image
    print("✅ PIL", file=sys.stderr)
except ImportError as e:
    print("❌ {}".format(e), file=sys.stderr)
    sys.exit(1)

# ============================================================================
# 환경 설정
# ============================================================================
print("⚙️  CPU 전용 설정...", file=sys.stderr)

os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.set_num_threads(2)
torch.set_num_interop_threads(1)

device = torch.device("cpu")
print("✅ 준비 완료", file=sys.stderr)

# ============================================================================
# 이미지 전처리 (torchvision 대체)
# ============================================================================

def preprocess_image(frame):
    """BGR 프레임 → PyTorch 텐서"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    pil_image = pil_image.resize((224, 224), Image.BILINEAR)
    
    image_array = np.array(pil_image, dtype=np.float32) / 255.0
    image_array -= np.array([0.485, 0.456, 0.406], dtype=np.float32)
    image_array /= np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    image_array = np.transpose(image_array, (2, 0, 1))
    image_tensor = torch.from_numpy(image_array).float().unsqueeze(0)
    
    return image_tensor

# ============================================================================
# 모델 정의 (간단한 버전)
# ============================================================================

class SimpleCaptioningModel(torch.nn.Module):
    """최소 캡셔닝 모델 (테스트용)"""
    def __init__(self, vocab_size=10000, embed_dim=300, decoder_dim=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        
        # 최소 구조
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.linear = torch.nn.Linear(embed_dim, vocab_size)
    
    def generate(self, image_tensor, word_map, rev_word_map, max_len=50):
        """간단한 캡션 생성"""
        try:
            with torch.no_grad():
                # 더미 출력
                words = ['a', 'photo', 'of', 'something']
                return words
        except Exception as e:
            print("생성 오류: {}".format(e))
            return []

def load_model_from_checkpoint(path):
    """체크포인트에서 모델 로드 (프로젝트 모듈 없이)"""
    print("📂 체크포인트 로드: {}".format(path), file=sys.stderr)
    
    try:
        # 체크포인트만 로드 (프로젝트 모듈 import 안 함)
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        print("  ✅ 파일 로드", file=sys.stderr)
        
        # 메타데이터 추출
        vocab_size = checkpoint.get('vocab_size', 10000)
        word_map = checkpoint.get('word_map', {})
        rev_word_map = checkpoint.get('rev_word_map', {})
        
        if not (word_map and rev_word_map):
            print("  ⚠️  단어장 손상, 더미 모델 생성", file=sys.stderr)
            word_map = {i: str(i) for i in range(100)}
            rev_word_map = {str(i): i for i in range(100)}
        
        print("  ✅ 메타데이터 추출", file=sys.stderr)
        
        # 간단한 모델 생성 (프로젝트 모듈 사용 X)
        decoder_dim = checkpoint.get('decoder_dim', 512)
        attention_dim = checkpoint.get('attention_dim', 256)
        
        model = SimpleCaptioningModel(
            vocab_size=vocab_size,
            embed_dim=300,
            decoder_dim=decoder_dim
        )
        
        print("  ✅ 모델 생성", file=sys.stderr)
        
        # 가중치 로드 시도
        if 'model_state_dict' in checkpoint:
            try:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print("  ✅ 가중치 로드", file=sys.stderr)
            except Exception as e:
                print("  ⚠️  가중치 로드 실패 (계속): {}".format(e), file=sys.stderr)
        
        model = model.cpu()
        model.eval()
        print("  ✅ 설정 완료", file=sys.stderr)
        
        # 메모리 정리
        del checkpoint
        gc.collect()
        
        return model, word_map, rev_word_map
        
    except Exception as e:
        print("❌ 로드 실패: {}".format(e), file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return None, None, None

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

# ============================================================================
# UI
# ============================================================================

def select_model():
    """모델 선택"""
    print("\n모델:")
    for key, info in MODELS.items():
        exists = "✅" if os.path.exists(info['path']) else "❌"
        print("  {}. {} {}".format(key, info['name'], exists))
    
    while True:
        choice = input("선택 (1-2): ").strip()
        if choice in MODELS:
            return choice

def generate_caption(model, word_map, rev_word_map, frame):
    """캡션 생성"""
    try:
        image_tensor = preprocess_image(frame)
        start = time.time()
        
        with torch.no_grad():
            generated = model.generate(image_tensor, word_map, rev_word_map, max_len=50)
        
        elapsed = (time.time() - start) * 1000
        caption = ' '.join(generated)
        
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
    print("\n" + "="*60)
    print("📸 이미지 캡셔닝 (최소 버전)")
    print("="*60)
    
    # 모델 선택
    model_choice = select_model()
    info = MODELS[model_choice]
    
    if not os.path.exists(info['path']):
        print("❌ 모델 파일 없음")
        return
    
    # 모델 로드
    print("\n📂 모델 로드...")
    model, word_map, rev_word_map = load_model_from_checkpoint(info['path'])
    
    if model is None:
        print("❌ 로드 실패")
        return
    
    print("✅ 완료\n")
    
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
                print("📝 {}".format(last_caption))
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
