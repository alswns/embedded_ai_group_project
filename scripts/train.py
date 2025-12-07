import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import re
import numpy as np
from collections import Counter, defaultdict
from src.muti_modal_model.model import MobileNetCaptioningModel
import warnings
from tqdm import tqdm

# 유틸리티 import
from src.utils import (
    setup_device,
    get_image_transform,
    CaptionDataset as CaptionDatasetUtil,  # 유틸 버전 (필요시 사용)
    calculate_meteor,
    METEOR_AVAILABLE,
)
from src.utils.glove_utils import (
    load_glove_embeddings_with_fallback,
    create_embedding_matrix
)
from src.utils.finetune_utils import (
    load_model_checkpoint,
    save_checkpoint as save_checkpoint_util,
)

warnings.filterwarnings("ignore")

# --- [0] 설정 (Configuration) ---
# 디바이스 선택: CUDA > MPS > CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA 사용 가능 - GPU 가속 활성화")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS 사용 가능 - Apple Silicon GPU 가속 활성화")
else:
    device = torch.device("cpu")
    print("CPU 모드로 실행")

LEARNING_RATE = 4e-4  # 학습률 (너무 크면 발산함)
BATCH_SIZE = 64 if device.type != "cpu" else 16  # GPU/MPS 사용 시 더 큰 배치
EPOCHS = 400          # 전체 반복 횟수
MAX_CAPTION_LEN = 50  # 최대 캡션 길이
MIN_WORD_FREQ = 2     # 단어장에 포함될 최소 빈도
ENCODER_FINE_TUNING = True
USE_MIXED_PRECISION = device.type in ["cuda", "mps"]  # Mixed precision (FP16) 사용
NUM_WORKERS = 0 if device.type == "mps" else 4  # MPS에서는 0이 안전, CUDA에서는 멀티프로세싱
PIN_MEMORY = device.type != "cpu"  # GPU 사용 시 메모리 고정

# === 경로 설정 (Colab 환경 자동 감지) ===
# Colab 환경 감지
IS_COLAB = 'COLAB_GPU' in os.environ or 'COLAB_TPU' in os.environ

if IS_COLAB:
    # Colab Google Drive 경로
    BASE_DIR = "/content/drive/MyDrive"
    IMAGES_DIR = os.path.join(BASE_DIR, "assets/images")
    CAPTIONS_FILE = os.path.join(BASE_DIR, "assets/captions.txt")
    MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models")
    ASSETS_DIR = os.path.join(BASE_DIR, "assets")
    
    print("🔵 Colab 환경 감지됨")
    print("   이미지 경로: {}".format(IMAGES_DIR))
    print("   캡션 파일: {}".format(CAPTIONS_FILE))
    print("   모델 저장 경로: {}".format(MODEL_SAVE_DIR))
    
    # 모델 저장 디렉토리 생성
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
else:
    # 로컬 환경
    IMAGES_DIR = "assets/images"
    CAPTIONS_FILE = "assets/captions.txt"
    MODEL_SAVE_DIR = "models"  # models 폴더에 저장
    ASSETS_DIR = "assets"
    print("🟢 로컬 환경")
    
    # 모델 저장 디렉토리 생성
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# 사전 학습된 임베딩 설정
EMBED_DIM = 300  # GloVe 6B.300d 사용 (또는 최적화 후 100으로 변경 가능)
USE_PRETRAINED_EMBEDDING = True  # 사전 학습된 임베딩 사용 여부

# GloVe 파일 경로 (assets 하위에 위치)
# 파일을 assets/glove.6B.300d.txt 위치에 저장
GLOVE_PATH = os.path.join(ASSETS_DIR, "glove.6B.300d.txt")
GLOVE_OPTIMIZED_PATH = os.path.join(ASSETS_DIR, "glove_optimized.pkl")

# --- [1] 이미지 전처리 ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])  # ImageNet 정규화
])

# --- [2] 캡션 전처리 함수 ---

def build_vocab(captions, min_freq=MIN_WORD_FREQ):
    """캡션 리스트로부터 단어장 생성"""
    # 모든 단어 수집
    word_counts = Counter()
    for caption in captions:
        # 소문자 변환 및 단어 분리
        words = re.findall(r'\w+', caption.lower())
        word_counts.update(words)
    
    # 최소 빈도 이상인 단어만 선택
    vocab = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3}
    idx = 4
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = idx
            idx += 1
    
    return vocab, {v: k for k, v in vocab.items()}

def encode_caption(caption, word_map, max_len=MAX_CAPTION_LEN):
    """캡션 텍스트를 정수 시퀀스로 변환"""
    words = re.findall(r'\w+', caption.lower())
    encoded = [word_map.get('<start>', 1)]
    
    for word in words:
        encoded.append(word_map.get(word, word_map.get('<unk>', 3)))
    
    encoded.append(word_map.get('<end>', 2))
    
    # 패딩 또는 자르기
    if len(encoded) > max_len:
        encoded = encoded[:max_len]
        encoded[-1] = word_map.get('<end>', 2)  # 마지막을 <end>로
    else:
        encoded.extend([word_map.get('<pad>', 0)] * (max_len - len(encoded)))
    
    return torch.tensor(encoded, dtype=torch.long)

# --- [3] 실제 데이터셋 클래스 ---
class CaptionDataset(Dataset):
    def __init__(self, images_dir, captions_file, transform=None, word_map=None, max_len=MAX_CAPTION_LEN):
        self.images_dir = images_dir
        self.transform = transform
        self.word_map = word_map
        self.max_len = max_len
        
        # 이미지 파일 목록 가져오기
        available_images = set([f for f in os.listdir(images_dir) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        
        # 캡션 파일 읽기 (CSV 형식 지원)
        # 이미지 파일명 -> 캡션 리스트 매핑
        image_to_captions = defaultdict(list)
        
        if os.path.exists(captions_file):
            with open(captions_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # 첫 번째 줄이 헤더인지 확인
                first_line = lines[0].strip() if lines else ""
                start_idx = 1 if first_line.lower().startswith('image') or first_line.lower().startswith('filename') else 0
                
                for line in lines[start_idx:]:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # CSV 형식 (쉼표로 구분)
                    if ',' in line:
                        parts = line.split(',', 1)  # 첫 번째 쉼표만으로 분리
                        if len(parts) == 2:
                            img_name = parts[0].strip()
                            caption = parts[1].strip()
                            if img_name and caption and img_name in available_images:
                                image_to_captions[img_name].append(caption)
                    # 탭으로 구분된 경우
                    elif '\t' in line:
                        parts = line.split('\t', 1)
                        if len(parts) == 2:
                            img_name = parts[0].strip()
                            caption = parts[1].strip()
                            if img_name and caption and img_name in available_images:
                                image_to_captions[img_name].append(caption)
                    # 단순 캡션만 있는 경우 (이미지 파일명 순서대로 매칭)
                    else:
                        # 이 경우는 나중에 처리
                        pass
        
        # 이미지-캡션 쌍 생성 (하나의 이미지에 여러 캡션이 있으면 각각 별도 샘플로)
        self.image_caption_pairs = []
        for img_name, captions in image_to_captions.items():
            if captions:
                # 각 캡션을 별도 샘플로 추가
                for caption in captions:
                    self.image_caption_pairs.append((img_name, caption))
        # 단순 캡션만 있는 경우 처리 (이미지 파일 순서대로 매칭)
        if not self.image_caption_pairs:
            image_files = sorted(available_images)
            if os.path.exists(captions_file):
                with open(captions_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    captions_only = [line.strip() for line in lines if line.strip() and not line.strip().startswith('image')]
                    for img_file, caption in zip(image_files, captions_only):
                        if caption:
                            self.image_caption_pairs.append((img_file, caption))
        
        print("로드된 데이터: {}개의 이미지-캡션 쌍".format(len(self.image_caption_pairs)))
        print("고유 이미지 수: {}".format(len(set([pair[0] for pair in self.image_caption_pairs]))))
        
    def __getitem__(self, idx):
        # 이미지 로드
        img_name, caption_text = self.image_caption_pairs[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print("이미지 로드 실패: {}, 오류: {}".format(img_path, e))
            # 오류 시 검은 이미지 반환
            image = torch.zeros(3, 224, 224)
        # 캡션 인코딩
        if self.word_map:
            caption = encode_caption(caption_text, self.word_map, self.max_len)
        else:
            # 단어장이 없으면 더미 캡션
            caption = torch.zeros(self.max_len, dtype=torch.long)
        
        return image, caption
    
    def __len__(self):
        return len(self.image_caption_pairs)

# --- [2] 학습 함수 정의 ---
def train_epoch(model, dataloader, criterion, optimizer, epoch, vocab_size, scaler=None, use_mixed_precision=False):
    model.train() # 학습 모드 설정
    total_loss = 0
    for i, (imgs, caps) in enumerate(tqdm(dataloader, desc="Training Epoch {}".format(epoch+1))):
        imgs = imgs.to(device, non_blocking=True)
        caps = caps.to(device, non_blocking=True)
        
        # 1. 기울기 초기화
        optimizer.zero_grad()
        
        # 2. Mixed Precision Training (FP16)
        if use_mixed_precision:
            if device.type == "cuda" and scaler is not None:
                with torch.cuda.amp.autocast():
                    # 모델 예측 (Forward)
                    outputs, alphas = model(imgs, caps)
                    
                    # 정답과 비교를 위한 차원 조절
                    targets = caps[:, 1:] 
                    outputs = outputs[:, :targets.shape[1], :]
                    
                    # 손실 계산
                    loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
                
                # 역전파 (Scaled)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            elif device.type == "mps":
                # MPS는 autocast만 지원
                with torch.amp.autocast(device_type="mps", dtype=torch.float16):
                    # 모델 예측 (Forward)
                    outputs, alphas = model(imgs, caps)
                    
                    # 정답과 비교를 위한 차원 조절
                    targets = caps[:, 1:] 
                    outputs = outputs[:, :targets.shape[1], :]
                    
                    # 손실 계산
                    loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
                
                # 역전파
                loss.backward()
                optimizer.step()
            else:
                # 일반 학습으로 폴백
                outputs, alphas = model(imgs, caps)
                targets = caps[:, 1:] 
                outputs = outputs[:, :targets.shape[1], :]
                loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
                loss.backward()
                optimizer.step()
        else:
            # 일반 학습 (FP32)
            # 모델 예측 (Forward)
            outputs, alphas = model(imgs, caps)
            
            # 정답과 비교를 위한 차원 조절
            targets = caps[:, 1:] 
            outputs = outputs[:, :targets.shape[1], :]
            
            # 손실 계산
            loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
            
            # 역전파
            loss.backward()
            
            # 가중치 업데이트
            optimizer.step()
        
        total_loss += loss.item()
        
        # if i % 10 == 0:
        #     print("Epoch [{}/{}], Step [{}/{}], Loss: {}".format(epoch+1, EPOCHS, i, len(dataloader), loss.item():.4f))
    return total_loss / len(dataloader)

# --- [2.5] 검증 함수 정의 ---
def validate_epoch(model, val_dataloader, criterion, epoch, vocab_size, word_map=None, rev_word_map=None):
    """검증 데이터셋에서 모델 평가 (Loss + METEOR 점수)"""
    model.eval()
    total_val_loss = 0
    meteor_scores = []
    
    with torch.no_grad():
        for i, (imgs, caps) in enumerate(val_dataloader):
            imgs = imgs.to(device, non_blocking=True)
            caps = caps.to(device, non_blocking=True)
            
            # 모델 예측 (Forward)
            outputs, alphas = model(imgs, caps)
            
            # 정답과 비교를 위한 차원 조절
            targets = caps[:, 1:] 
            outputs = outputs[:, :targets.shape[1], :]
            
            # 손실 계산
            loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
            total_val_loss += loss.item()
            
            # METEOR 점수 계산 (word_map이 제공된 경우)
            if word_map is not None and rev_word_map is not None:
                try:
                    # 배치의 각 샘플에 대해 캡션 생성
                    for j in range(imgs.shape[0]):
                        img_single = imgs[j:j+1]
                        cap_single = caps[j:j+1]
                        
                        # 캡션 생성
                        generated_words = model.generate(img_single, word_map, rev_word_map, max_len=MAX_CAPTION_LEN)
                        generated_caption = ' '.join([w for w in generated_words if w not in ['<start>', '<end>', '<pad>', '<unk>']])
                        
                        # 참조 캡션
                        reference_cap = ' '.join([rev_word_map.get(int(idx), '<unk>') for idx in cap_single[0] if int(idx) > 0])
                        reference_cap = reference_cap.replace('<start> ', '').replace(' <end>', '')
                        
                        # METEOR 계산 (유틸 함수 사용)
                        meteor = calculate_meteor(
                            generated_caption.lower().split(),
                            reference_cap
                        )
                        if meteor is None:
                            meteor = 0.0
                        
                        meteor_scores.append(meteor)
                except Exception as e:
                    # METEOR 계산 실패 시 0.0 추가
                    meteor_scores.append(0.0)
            
            if i % 10 == 0:
                print("  Validation Step [{}/{}], Loss: {:.4f}".format(i, len(val_dataloader), loss.item()))
    
    avg_val_loss = total_val_loss / len(val_dataloader)
    avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0
    
    model.train()  # 다시 학습 모드로
    
    return avg_val_loss, avg_meteor


def evaluate_multiple_samples(model, dataset, word_map, rev_word_map, num_samples=5, start_idx=0):
    """val 데이터셋 전체의 평균 METEOR 점수 계산"""
    model.eval()
    
    meteor_scores = []
    
    # 전체 val 데이터셋 사용 (num_samples는 무시)
    total_samples = len(dataset)
    
    print("\n{'='*70}")
    print("🔍 검증 데이터셋 평가: {}개 샘플의 평균 METEOR 계산".format(total_samples))
    print("{'='*70}")
    
    with torch.no_grad():
        for i in range(total_samples):
            try:
                img_name, original_caption = dataset.image_caption_pairs[i]
                image, _ = dataset[i]
                
                # 배치 차원 추가 [1, 3, 224, 224]
                image = image.unsqueeze(0).to(device)
                
                # 캡션 생성
                generated_words = model.generate(image, word_map, rev_word_map, max_len=MAX_CAPTION_LEN)
                
                # 토큰 제거하고 문장으로 변환
                generated_caption = ' '.join([w for w in generated_words if w not in ['<start>', '<end>', '<pad>', '<unk>']])
                
                # METEOR 점수 계산 (유틸 함수 사용)
                meteor = calculate_meteor(
                    generated_caption.lower().split(),
                    original_caption
                )
                if meteor is None:
                    meteor = 0.0
                
                meteor_scores.append(meteor)
                
                # 진행도 표시 (100개마다)
                if (i + 1) % 100 == 0:
                    current_avg = sum(meteor_scores) / len(meteor_scores)
                    print("  진행: {}/{}, 현재 평균 METEOR: {}".format(i+1, total_samples, current_avg:.4f))
                    
            except Exception as e:
                print("  ⚠️ 샘플 {} 생성 실패: {}".format(i+1, e))
                meteor_scores.append(0.0)
    
    # 전체 평균 METEOR 점수
    avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0
    
    print("\n{'='*70}")
    print("📈 검증 데이터셋 METEOR 통계:")
    print("  • 평가 샘플: {}개".format(total_samples))
    print("  • 평균 METEOR 점수: {}".format(avg_meteor:.4f))
    if meteor_scores:
        print("  • 최고 METEOR 점수: {}".format(max(meteor_scores):.4f))
        print("  • 최저 METEOR 점수: {}".format(min(meteor_scores):.4f))
        print("  • METEOR 점수 분포:")
        print("    - 0.5 이상 (우수): {}개".format(sum([1 for s in meteor_scores if s >= 0.5])))
        print("    - 0.3-0.5 (양호): {}개".format(sum([1 for s in meteor_scores if 0.3 <= s < 0.5])))
        print("    - 0.3 미만 (개선 필요): {}개".format(sum([1 for s in meteor_scores if s < 0.3])))
    print("{'='*70}\n")
    
    model.train()  # 다시 학습 모드로
    
    return {
        'avg_meteor': avg_meteor,
        'max_meteor': max(meteor_scores) if meteor_scores else 0.0,
        'min_meteor': min(meteor_scores) if meteor_scores else 0.0,
        'meteor_scores': meteor_scores
    }

# --- [4] 메인 실행 코드 ---
def main():
    # 1. 캡션 파일 읽어서 단어장 생성
    print("단어장 생성 중...")
    captions_list = []
    if os.path.exists(CAPTIONS_FILE):
        with open(CAPTIONS_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # 첫 번째 줄이 헤더인지 확인
            first_line = lines[0].strip() if lines else ""
            start_idx = 1 if first_line.lower().startswith('image') or first_line.lower().startswith('filename') else 0
            
            for line in lines[start_idx:]:
                line = line.strip()
                if line:
                    # CSV 형식 (쉼표로 구분)
                    if ',' in line:
                        parts = line.split(',', 1)
                        if len(parts) == 2:
                            captions_list.append(parts[1].strip())
                    # 탭으로 구분된 경우 캡션 부분만 추출
                    elif '\t' in line:
                        parts = line.split('\t', 1)
                        captions_list.append(parts[1] if len(parts) > 1 else parts[0])
                    else:
                        captions_list.append(line)
    
    if not captions_list:
        print("경고: 캡션 파일이 비어있거나 없습니다. 더미 데이터를 사용합니다.")
        captions_list = ["a cat sitting on a mat"] * 100  # 더미 데이터
    
    word_map, rev_word_map = build_vocab(captions_list, min_freq=MIN_WORD_FREQ)
    vocab_size = len(word_map)
    print("단어장 크기: {}".format(vocab_size))
    print("주요 단어 예시: {}".format(list(word_map.items())[:10]))
    
    # 사전 학습된 임베딩 로드 (유틸 함수 사용)
    use_pretrained = USE_PRETRAINED_EMBEDDING
    glove_embeddings = None
    actual_embed_dim = EMBED_DIM
    
    if use_pretrained:
        glove_embeddings, actual_embed_dim = load_glove_embeddings_with_fallback(
            GLOVE_PATH, GLOVE_OPTIMIZED_PATH, EMBED_DIM
        )
        if glove_embeddings is None:
            print("⚠️ 사전 학습된 임베딩을 사용할 수 없습니다. 랜덤 초기화를 사용합니다.")
            use_pretrained = False
    
    # 임베딩 행렬 생성
    embedding_matrix = None
    if use_pretrained and glove_embeddings:
        embedding_matrix = create_embedding_matrix(word_map, glove_embeddings, embed_dim=actual_embed_dim)
    else:
        # 랜덤 초기화 사용 시에도 embed_dim은 설정값 사용
        pass
    
    # 2. 데이터셋 및 데이터 로더 준비
    print("데이터셋 로드 중...")
    dataset = CaptionDataset(
        images_dir=IMAGES_DIR,
        captions_file=CAPTIONS_FILE,
        transform=transform,
        word_map=word_map,
        max_len=MAX_CAPTION_LEN
    )
    
    if len(dataset) == 0:
        raise ValueError("데이터셋이 비어있습니다. {} 폴더에 이미지가 있는지 확인하세요.".format(IMAGES_DIR))
    
    # 검증 셋 분리 (80% 학습, 20% 검증)
    val_split_ratio = 0.1
    val_size = max(1, int(len(dataset) * val_split_ratio))
    train_size = len(dataset) - val_size
    
    # 시드 고정으로 재현성 보장
    torch.manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size]
    )
    
    print("✅ 데이터셋 분할 완료:")
    print("   • 학습 셋: {}개 샘플".format(len(train_dataset)))
    print("   • 검증 셋: {}개 샘플".format(len(val_dataset)))
    
    # 최적화된 DataLoader 설정
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None
    )
    
    # 검증 데이터 로더 (셔플 불필요)
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None
    )
    
    # 3. 모델 준비 (MobileNet + Decoder)
    print("모델 초기화 중...")
    model = MobileNetCaptioningModel(vocab_size=vocab_size, embed_dim=EMBED_DIM).to(device)
    
    # 사전 학습된 임베딩 가중치 설정
    if use_pretrained and embedding_matrix is not None:
        print("사전 학습된 임베딩 가중치 설정 중...")
        model.decoder.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        # 임베딩을 학습 가능하게 할지 고정할지 선택 (True: 학습, False: 고정)
        model.decoder.embedding.weight.requires_grad = True
        print("✅ 사전 학습된 임베딩 가중치 설정 완료")
    
    # 체크포인트에서 모델 로드 (있는 경우)
    checkpoint_path = os.path.join(MODEL_SAVE_DIR, "lightweight_captioning_model.pth")
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print("📂 체크포인트 발견: {}".format(checkpoint_path))
        try:
            # Python/PyTorch 버전 호환성
            try:
                # Python 3.11+: weights_only 파라미터 필요
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            except TypeError:
                # Python 3.6-3.10: weights_only 파라미터 미지원
                checkpoint = torch.load(checkpoint_path, map_location=device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                start_epoch = checkpoint.get('epoch', 0)
                print("✅ 체크포인트에서 모델 로드 완료 (Epoch {}부터 이어서 학습)".format(start_epoch))
            else:
                # 딕셔너리가 아닌 경우 (구버전 체크포인트)
                model.load_state_dict(checkpoint)
                print("✅ 체크포인트에서 모델 로드 완료")
        except Exception as e:
            print("⚠️ 체크포인트 로드 실패: {}".format(e))
            print("   새로 학습을 시작합니다.")
    else:
        print("📝 체크포인트 없음 - 새로 학습 시작")
    
    model.to(device)
    
    # [핵심] 4. 인코더 얼리기 (Encoder Freezing)
    # MobileNet 부분은 학습되지 않도록 설정 (이미지넷 지식 보존)
    for param in model.encoder.parameters():
        param.requires_grad = False
        
    # 5. 최적화 도구 설정
    # filter를 써서 requires_grad=True인 파라미터(디코더)만 업데이트 목록에 넣음
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    # 스케줄러: METEOR 점수 기반으로 학습률 동적 조정
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max',           # METEOR(max)이 기준 (METEOR가 높을수록 좋음)
        factor=0.66,          # 학습률을 0.66배 감소
        patience=2,           # 2 에포크 동안 개선 없으면 학습률 감소
        min_lr=1e-6           # 최소 학습률
    )
    # 6. 손실 함수 (Padding=0 무시)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # 7. Mixed Precision Scaler 설정
    scaler = None
    use_mixed_precision = USE_MIXED_PRECISION  # 로컬 변수로 복사

    if use_mixed_precision:
        if device.type == "cuda":
            scaler = torch.cuda.amp.GradScaler()
            print("Mixed Precision (FP16) 활성화 - CUDA")
        elif device.type == "mps":
            # MPS는 autocast만 지원하고 GradScaler는 없음
            print("Mixed Precision (FP16) 활성화 - MPS")
        else:
            use_mixed_precision = False
    
    # 8. 학습 루프
    print("학습 시작 (Encoder Frozen)... 총 {}개 샘플, {} 에포크".format(len(train_dataset), EPOCHS))
    print("배치 크기: {}, 디바이스: {}, Mixed Precision: {}".format(BATCH_SIZE, device, use_mixed_precision))
    
    # 검증 설정
    VAL_NUM_SAMPLES = max(5, len(val_dataset))  # 검증에 사용할 샘플 수
    
    # 학습 이력 추적
    train_losses = []
    val_losses = []
    
    # 체크포인트에서 이어서 학습하는 경우
    for epoch in range(start_epoch, EPOCHS):
        print("\n{'='*70}")
        print("Epoch {}/{} 시작".format(epoch+1, EPOCHS))
        print("{'='*70}")
        
        # 학습 에포크
        avg_train_loss = train_epoch(model, train_dataloader, criterion, optimizer, epoch, vocab_size, scaler, use_mixed_precision)
        train_losses.append(avg_train_loss)
        print("✅ 학습 완료. 평균 Loss: {}".format(avg_train_loss:.4f))
        
        # 검증 에포크 (Loss + METEOR 점수 계산)
        print("\n🔍 검증 시작...")
        avg_val_loss, avg_meteor = validate_epoch(
            model, val_dataloader, criterion, epoch, vocab_size, 
            word_map=word_map, rev_word_map=rev_word_map
        )
        val_losses.append(avg_val_loss)
        print("✅ 검증 완료. 평균 Loss: {}".format(avg_val_loss:.4f))
        print("⭐ 평균 METEOR: {}".format(avg_meteor:.4f))
        
        # 스케줄러 업데이트 (METEOR 점수 기반)
        scheduler.step(avg_meteor)
        current_lr = optimizer.param_groups[0]['lr']
        print("📊 스케줄러 업데이트 - METEOR: {}, Learning Rate: {}".format(avg_meteor:.4f, current_lr:.2e))
        
        # [옵션] 특정 Epoch 이후에 인코더도 같이 학습시키고 싶다면? (Fine-tuning)
        if ENCODER_FINE_TUNING and epoch == 5:
            print(">>> 인코더 미세 조정 시작 (Fine-tuning Start) <<<")
            # 인코더의 뒷부분 레이어만 풀거나 전체를 풂
            for param in model.encoder.parameters():
                param.requires_grad = True
            
            # 옵티마이저에 인코더 파라미터도 추가 (학습률은 더 낮게 잡는 게 좋음)
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE * 0.1)
            scheduler = ReduceLROnPlateau(
                optimizer, 
                mode='max',      # METEOR(max)이 기준
                factor=0.66, 
                patience=2,
            )

        # 주기적으로 모델 저장
        save_path = os.path.join(MODEL_SAVE_DIR, "lightweight_captioning_model_{}_epoch_meteor_{}.pth".format(epoch+1, avg_meteor:.4f))
        try:
            torch.save({
                'model_state_dict': model.state_dict(),
                'word_map': word_map,
                'rev_word_map': rev_word_map,
                'vocab_size': vocab_size,
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }, save_path)
            print("✅ 모델 저장 완료: {}".format(save_path))
        except Exception as e:
            print("❌ 모델 저장 실패: {}".format(e))
            print("   저장 경로: {}".format(save_path))
        
        print("{'='*70}\n")
    
    # 8. 최종 모델 저장
    final_save_path = os.path.join(MODEL_SAVE_DIR, "lightweight_captioning_model.pth")
    try:
        torch.save({
            'model_state_dict': model.state_dict(),
            'word_map': word_map,
            'rev_word_map': rev_word_map,
            'vocab_size': vocab_size,
            'epoch': EPOCHS,
            'train_losses': train_losses,
            'val_losses': val_losses
        }, final_save_path)
        print("\n✅ 최종 모델 저장 완료: {}".format(final_save_path))
        
        # 학습 통계 출력
        print("\n{'='*70}")
        print("📊 학습 완료 통계:")
        print("{'='*70}")
        print("  • 최종 학습 손실: {}".format(train_losses[-1]:.4f))
        print("  • 최종 검증 손실: {}".format(val_losses[-1]:.4f))
        print("  • 최소 검증 손실: {} (Epoch {})".format(min(val_losses):.4f, val_losses.index(min(val_losses))+1))
        print("  • 학습 손실 개선도: {}%".format(((train_losses[0]-train_losses[-1])/train_losses[0]*100):.2f))
        print("  • 검증 손실 개선도: {}%".format(((val_losses[0]-val_losses[-1])/val_losses[0]*100):.2f))
        print("{'='*70}\n")
    except Exception as e:
        print("❌ 최종 모델 저장 실패: {}".format(e))
        print("   저장 경로: {}".format(final_save_path))

if __name__ == "__main__":
    main()