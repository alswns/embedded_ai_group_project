import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import re
import random
import numpy as np
from collections import Counter, defaultdict
from src.muti_modal_model.model import MobileNetCaptioningModel

# METEOR 점수 계산을 위한 nltk
try:
    from nltk.translate.meteor_score import meteor_score
    from nltk.tokenize import word_tokenize
    import nltk
    # 필요한 데이터 다운로드
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    METEOR_AVAILABLE = True
except ImportError:
    print("⚠️ nltk가 설치되지 않았습니다. METEOR 점수를 계산할 수 없습니다.")
    print("   설치: pip install nltk")
    METEOR_AVAILABLE = False
    meteor_score = None

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

LEARNING_RATE = 2e-4  # 학습률 (너무 크면 발산함)
BATCH_SIZE = 64 if device.type != "cpu" else 16  # GPU/MPS 사용 시 더 큰 배치
EPOCHS = 100          # 전체 반복 횟수
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
    
    print(f"🔵 Colab 환경 감지됨")
    print(f"   이미지 경로: {IMAGES_DIR}")
    print(f"   캡션 파일: {CAPTIONS_FILE}")
    print(f"   모델 저장 경로: {MODEL_SAVE_DIR}")
    
    # 모델 저장 디렉토리 생성
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
else:
    # 로컬 환경
    IMAGES_DIR = "assets/images"
    CAPTIONS_FILE = "assets/captions.txt"
    MODEL_SAVE_DIR = "models"  # models 폴더에 저장
    ASSETS_DIR = "assets"
    print(f"🟢 로컬 환경")
    
    # 모델 저장 디렉토리 생성
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# 사전 학습된 임베딩 설정
EMBED_DIM = 300  # GloVe 6B.300d 사용
USE_PRETRAINED_EMBEDDING = True  # 사전 학습된 임베딩 사용 여부
# GloVe 파일 경로 (assets 하위에 위치)
# 다운로드: wget http://nlp.stanford.edu/data/glove.6B.zip && unzip glove.6B.zip
# 파일을 assets/glove.6B.300d.txt 위치에 저장
GLOVE_PATH = os.path.join(ASSETS_DIR, "glove.6B.300d.txt")

# --- [1] 이미지 전처리 ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])  # ImageNet 정규화
])

# --- [2] 캡션 전처리 함수 ---
def load_glove_embeddings(glove_path, embed_dim=300):
    """GloVe 임베딩 파일 로드"""
    print(f"GloVe 임베딩 로드 중: {glove_path}")
    embeddings_dict = {}
    
    if not os.path.exists(glove_path):
        print(f"⚠️ GloVe 파일을 찾을 수 없습니다: {glove_path}")
        print("\n📥 GloVe 다운로드 방법:")
        print("  방법 1 (터미널):")
        print(f"    wget http://nlp.stanford.edu/data/glove.6B.zip")
        print(f"    unzip glove.6B.zip")
        print(f"    mv glove.6B.300d.txt {ASSETS_DIR}/")
        print("  방법 2 (Colab):")
        print(f"    !wget http://nlp.stanford.edu/data/glove.6B.zip")
        print(f"    !unzip glove.6B.zip")
        print(f"    !mv glove.6B.300d.txt {ASSETS_DIR}/")
        print("  방법 3 (수동):")
        print("    https://nlp.stanford.edu/projects/glove/ 에서 다운로드")
        print(f"    다운로드한 glove.6B.300d.txt 파일을 {ASSETS_DIR}/ 폴더에 저장")
        print(f"\n💡 GloVe 파일이 없으면 랜덤 초기화된 임베딩을 사용합니다.")
        print(f"   예상 경로: {glove_path}\n")
        return None
    
    try:
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                if len(vector) == embed_dim:
                    embeddings_dict[word] = vector
        
        print(f"✅ GloVe 임베딩 로드 완료: {len(embeddings_dict)}개 단어")
        return embeddings_dict
    except Exception as e:
        print(f"⚠️ GloVe 로드 실패: {e}")
        return None

def create_embedding_matrix(word_map, glove_embeddings=None, embed_dim=300):
    """단어장에 맞는 임베딩 행렬 생성"""
    vocab_size = len(word_map)
    embedding_matrix = np.random.normal(scale=0.6, size=(vocab_size, embed_dim))
    
    if glove_embeddings is None:
        print("⚠️ 사전 학습된 임베딩 없음 - 랜덤 초기화 사용")
        return embedding_matrix
    
    # 특수 토큰은 랜덤 초기화 유지
    found_count = 0
    for word, idx in word_map.items():
        if word in ['<pad>', '<start>', '<end>', '<unk>']:
            continue  # 특수 토큰은 랜덤 초기화 유지
        
        if word in glove_embeddings:
            embedding_matrix[idx] = glove_embeddings[word]
            found_count += 1
        elif word.lower() in glove_embeddings:
            embedding_matrix[idx] = glove_embeddings[word.lower()]
            found_count += 1
    
    print(f"✅ 임베딩 행렬 생성 완료: {found_count}/{vocab_size-4}개 단어 매칭 (특수 토큰 제외)")
    return embedding_matrix

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
        
        print(f"로드된 데이터: {len(self.image_caption_pairs)}개의 이미지-캡션 쌍")
        print(f"고유 이미지 수: {len(set([pair[0] for pair in self.image_caption_pairs]))}")
        
    def __getitem__(self, idx):
        # 이미지 로드
        img_name, caption_text = self.image_caption_pairs[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"이미지 로드 실패: {img_path}, 오류: {e}")
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
    
    for i, (imgs, caps) in enumerate(dataloader):
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
        
        if i % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}")
    return total_loss / len(dataloader)

# --- [3] 여러 샘플로 캡션 생성 및 검증 출력 ---
def evaluate_multiple_samples(model, dataset, word_map, rev_word_map, num_samples=5, start_idx=0):
    """여러 샘플 이미지로 캡션을 생성하고 METEOR 점수로 검증"""
    model.eval()
    
    results = []
    meteor_scores = []
    
    print(f"\n{'='*70}")
    print(f"🔍 검증: {num_samples}개 샘플로 캡션 생성 및 METEOR 평가")
    print(f"{'='*70}")
    
    with torch.no_grad():
        for i in range(num_samples):
            idx = (start_idx + i) % len(dataset)
            
            img_name, original_caption = dataset.image_caption_pairs[idx]
            image, _ = dataset[idx]
            
            # 이미지 파일 전체 경로
            img_path = os.path.join(dataset.images_dir, img_name)
            
            # 배치 차원 추가 [1, 3, 224, 224]
            image = image.unsqueeze(0).to(device)
            
            try:
                # 캡션 생성
                generated_words = model.generate(image, word_map, rev_word_map, max_len=MAX_CAPTION_LEN)
                
                # 토큰 제거하고 문장으로 변환
                generated_caption = ' '.join([w for w in generated_words if w not in ['<start>', '<end>', '<pad>', '<unk>']])
                
                # METEOR 점수 계산
                meteor = 0.0
                if METEOR_AVAILABLE and meteor_score:
                    try:
                        # METEOR는 reference를 리스트로 받음 (여러 참조 가능)
                        reference = [original_caption.lower().split()]
                        hypothesis = generated_caption.lower().split()
                        meteor = meteor_score(reference, hypothesis)
                    except Exception as e:
                        # METEOR 계산 실패 시 단어 일치율로 대체
                        original_words = set(original_caption.lower().split())
                        generated_words_set = set(generated_caption.lower().split())
                        common_words = original_words & generated_words_set
                        meteor = len(common_words) / len(original_words) if len(original_words) > 0 else 0.0
                else:
                    # nltk가 없으면 단어 일치율로 대체
                    original_words = set(original_caption.lower().split())
                    generated_words_set = set(generated_caption.lower().split())
                    common_words = original_words & generated_words_set
                    meteor = len(common_words) / len(original_words) if len(original_words) > 0 else 0.0
                
                meteor_scores.append(meteor)
                
                results.append({
                    'img_name': img_name,
                    'original': original_caption,
                    'generated': generated_caption,
                    'meteor': meteor
                })
                
                # 각 샘플 출력
                print(f"\n[샘플 {i+1}/{num_samples}]")
                print(f"  📸 이미지: {img_name}")
                print(f"  📝 원본: {original_caption}")
                print(f"  🤖 생성: {generated_caption}")
                print(f"  ⭐ METEOR: {meteor:.4f}")
                
            except Exception as e:
                print(f"  ⚠️ 샘플 {i+1} 생성 실패: {e}")
                meteor_scores.append(0.0)
                results.append({
                    'img_name': img_name,
                    'original': original_caption,
                    'generated': '생성 실패',
                    'meteor': 0.0
                })
    
    # 전체 통계 출력
    avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0
    good_results = sum([1 for score in meteor_scores if score > 0.3])  # 0.3 이상을 좋은 결과로 간주
    
    print(f"\n{'='*70}")
    print(f"📈 METEOR 검증 통계:")
    print(f"  • 평균 METEOR 점수: {avg_meteor:.4f}")
    print(f"  • 최고 METEOR 점수: {max(meteor_scores):.4f}")
    print(f"  • 최저 METEOR 점수: {min(meteor_scores):.4f}")
    print(f"  • 좋은 결과 비율: {good_results}/{num_samples} ({good_results/num_samples*100:.1f}%)")
    print(f"  • METEOR 점수 분포:")
    print(f"    - 0.5 이상 (우수): {sum([1 for s in meteor_scores if s >= 0.5])}개")
    print(f"    - 0.3-0.5 (양호): {sum([1 for s in meteor_scores if 0.3 <= s < 0.5])}개")
    print(f"    - 0.3 미만 (개선 필요): {sum([1 for s in meteor_scores if s < 0.3])}개")
    print(f"{'='*70}\n")
    
    model.train()  # 다시 학습 모드로
    
    return {
        'avg_meteor': avg_meteor,
        'max_meteor': max(meteor_scores) if meteor_scores else 0.0,
        'min_meteor': min(meteor_scores) if meteor_scores else 0.0,
        'good_results': good_results / num_samples if num_samples > 0 else 0.0
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
    print(f"단어장 크기: {vocab_size}")
    print(f"주요 단어 예시: {list(word_map.items())[:10]}")
    
    # 사전 학습된 임베딩 로드
    use_pretrained = USE_PRETRAINED_EMBEDDING  # 로컬 변수로 복사
    glove_embeddings = None
    if use_pretrained:
        glove_embeddings = load_glove_embeddings(GLOVE_PATH, embed_dim=EMBED_DIM)
        if glove_embeddings is None:
            print("⚠️ 사전 학습된 임베딩을 사용할 수 없습니다. 랜덤 초기화를 사용합니다.")
            use_pretrained = False
    
    # 임베딩 행렬 생성
    embedding_matrix = None
    if use_pretrained and glove_embeddings:
        embedding_matrix = create_embedding_matrix(word_map, glove_embeddings, embed_dim=EMBED_DIM)
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
        raise ValueError(f"데이터셋이 비어있습니다. {IMAGES_DIR} 폴더에 이미지가 있는지 확인하세요.")
    
    # 최적화된 DataLoader 설정
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
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
        print(f"📂 체크포인트 발견: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                start_epoch = checkpoint.get('epoch', 0)
                print(f"✅ 체크포인트에서 모델 로드 완료 (Epoch {start_epoch}부터 이어서 학습)")
            else:
                # 딕셔너리가 아닌 경우 (구버전 체크포인트)
                model.load_state_dict(checkpoint)
                print(f"✅ 체크포인트에서 모델 로드 완료")
        except Exception as e:
            print(f"⚠️ 체크포인트 로드 실패: {e}")
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
    print(f"학습 시작 (Encoder Frozen)... 총 {len(dataset)}개 샘플, {EPOCHS} 에포크")
    print(f"배치 크기: {BATCH_SIZE}, 디바이스: {device}, Mixed Precision: {use_mixed_precision}")
    
    # 검증 설정
    VAL_NUM_SAMPLES = 5  # 검증에 사용할 샘플 수
    val_start_idx = 0  # 검증 시작 인덱스 (매 epoch마다 변경 가능)
    
    # 체크포인트에서 이어서 학습하는 경우
    for epoch in range(start_epoch, EPOCHS):
        avg_loss = train_epoch(model, dataloader, criterion, optimizer, epoch, vocab_size, scaler, use_mixed_precision)
        print(f"=== Epoch {epoch+1}/{EPOCHS} 완료. 평균 Loss: {avg_loss:.4f} ===")
        
        # 여러 샘플로 검증 및 출력
        val_results = evaluate_multiple_samples(
            model, dataset, word_map, rev_word_map, 
            num_samples=VAL_NUM_SAMPLES, 
            start_idx=(val_start_idx + epoch * VAL_NUM_SAMPLES) % len(dataset)
        )
        
        # [옵션] 특정 Epoch 이후에 인코더도 같이 학습시키고 싶다면? (Fine-tuning)
        if ENCODER_FINE_TUNING and epoch == 5:
            print(">>> 인코더 미세 조정 시작 (Fine-tuning Start) <<<")
            # 인코더의 뒷부분 레이어만 풀거나 전체를 풂
            for param in model.encoder.parameters():
                param.requires_grad = True
            
            # 옵티마이저에 인코더 파라미터도 추가 (학습률은 더 낮게 잡는 게 좋음)
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE * 0.1)

        # 주기적으로 모델 저장
        save_path = os.path.join(MODEL_SAVE_DIR, f"lightweight_captioning_model_{epoch+1}_epoch.pth")
        try:
            torch.save({
                'model_state_dict': model.state_dict(),
                'word_map': word_map,
                'rev_word_map': rev_word_map,
                'vocab_size': vocab_size,
                'epoch': epoch + 1
            }, save_path)
            print(f"✅ 모델 저장 완료: {save_path}")
        except Exception as e:
            print(f"❌ 모델 저장 실패: {e}")
            print(f"   저장 경로: {save_path}")
    
    # 8. 최종 모델 저장
    final_save_path = os.path.join(MODEL_SAVE_DIR, "lightweight_captioning_model.pth")
    try:
        torch.save({
            'model_state_dict': model.state_dict(),
            'word_map': word_map,
            'rev_word_map': rev_word_map,
            'vocab_size': vocab_size,
            'epoch': EPOCHS
        }, final_save_path)
        print(f"✅ 최종 모델 저장 완료: {final_save_path}")
    except Exception as e:
        print(f"❌ 최종 모델 저장 실패: {e}")
        print(f"   저장 경로: {final_save_path}")

if __name__ == "__main__":
    main()