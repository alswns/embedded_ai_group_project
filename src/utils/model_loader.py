"""
모델 로드 함수
"""
import torch
import torch.nn as nn
import os
from .config import MODEL_PATH

# 모델 import
try:
    from src.muti_modal_model.model import MobileNetCaptioningModel
except ImportError:
    print("⚠️ 모델 클래스를 import할 수 없습니다. 경로를 확인해주세요.")
    class MobileNetCaptioningModel(nn.Module):
        def __init__(self, vocab_size, embed_dim):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, embed_dim)
            self.gru = nn.GRU(embed_dim, 512)
            self.fc = nn.Linear(512, vocab_size)
        def generate(self, img, wm, rwm, max_len):
            return ["<start>", "a", "test", "caption", "<end>"]

def load_base_model(model_path=None, device=None):
    """학습된 모델 로드
    
    Args:
        model_path: 모델 파일 경로 (None이면 기본 경로 사용)
        device: torch device (None이면 CPU 사용)
    
    Returns:
        model, word_map, rev_word_map
    """
    if model_path is None:
        model_path = MODEL_PATH
    if device is None:
        device = torch.device("cpu")
    
    print("📂 모델 로드 중...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 체크포인트에서 정보 추출
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            vocab_size = checkpoint.get('vocab_size', 1000)
            word_map = checkpoint.get('word_map', {})
            rev_word_map = checkpoint.get('rev_word_map', {})
        else:
            model_state = checkpoint
            vocab_size = 1000
            word_map = {}
            rev_word_map = {}
    else:
        model_state = checkpoint
        vocab_size = 1000
        word_map = {}
        rev_word_map = {}
    
    # 모델 생성
    embed_dim = 300  # GloVe 사용 시
    model = MobileNetCaptioningModel(vocab_size=vocab_size, embed_dim=embed_dim)
    model.load_state_dict(model_state)
    model.eval()
    model.to(device)
    
    print(f"✅ 모델 로드 완료 (Vocab Size: {vocab_size})")
    return model, word_map, rev_word_map

