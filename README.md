# 이미지 캡셔닝 모델 (Image Captioning Model)

경량화된 이미지 캡셔닝 모델로, MobileNetV3 기반 인코더와 GRU 기반 디코더를 사용하여 이미지로부터 자연어 캡션을 생성합니다. 임베디드 장치(Jetson Nano) 및 다양한 환경(Mac, Colab)에서 실행 가능합니다.

## 📋 목차

- [주요 기능](#주요-기능)
- [프로젝트 구조](#프로젝트-구조)
- [설치 방법](#설치-방법)
- [데이터 준비](#데이터-준비)
- [학습 방법](#학습-방법)
- [사용 방법](#사용-방법)
- [Jetson Nano 설정](#jetson-nano-설정)
- [Google Colab 사용](#google-colab-사용)
- [설정 옵션](#설정-옵션)

## ✨ 주요 기능

- **경량화된 모델**: MobileNetV3 Small 기반으로 임베디드 장치에서도 실행 가능
- **사전 학습된 임베딩**: GloVe 6B.300d 워드 임베딩 지원
- **멀티 플랫폼**: Mac (MPS), CUDA GPU, CPU, Jetson Nano 지원
- **Mixed Precision**: FP16 학습으로 메모리 사용량 감소 및 속도 향상
- **자동 환경 감지**: Colab, 로컬 환경 자동 감지 및 경로 설정
- **검증 기능**: 각 epoch마다 여러 샘플로 캡션 생성 및 평가

## 📁 프로젝트 구조

```
임베디드/
├── assets/
│   ├── images/              # 학습용 이미지 파일들
│   ├── captions.txt         # 캡션 파일 (CSV 형식: image,caption)
│   └── glove.6B.300d.txt    # GloVe 임베딩 파일 (선택사항)
├── scripts/
│   ├── train.py            # 학습 스크립트
│   └── run.py              # 추론 테스트 스크립트
├── src/
│   ├── muti_modal_model/
│   │   └── model.py        # MobileNet + GRU 디코더 모델
│   ├── gru_model/
│   │   └── model.py        # GRU 기반 디코더
│   └── image_net/
│       └── model.py        # MobileNet 인코더
├── requirements.txt        # 패키지 의존성
└── README.md              # 이 파일
```

## 🚀 설치 방법

### 1. 저장소 클론

```bash
git clone <your-repo-url>
cd 임베디드
```

### 2. 가상환경 생성 (권장)

```bash
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# 또는
venv\Scripts\activate  # Windows
```

### 3. 패키지 설치

#### 기본 패키지

```bash
pip install -r requirements.txt
```

#### PyTorch 설치

**Mac (Apple Silicon):**
```bash
pip install torch torchvision
```

**Linux/Windows (CUDA):**
```bash
# CUDA 11.8 예시
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**CPU만 사용:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## 📦 데이터 준비

### 1. 이미지 파일 준비

`assets/images/` 폴더에 이미지 파일들을 저장합니다.

```bash
assets/images/
├── image1.jpg
├── image2.jpg
└── ...
```

### 2. 캡션 파일 준비

`assets/captions.txt` 파일을 CSV 형식으로 작성합니다.

**형식 1: CSV (권장)**
```
image,caption
image1.jpg,A child in a pink dress is climbing up stairs
image2.jpg,A dog playing in the park
```

**형식 2: 탭 구분**
```
image1.jpg	A child in a pink dress is climbing up stairs
image2.jpg	A dog playing in the park
```

**형식 3: 순서대로 (이미지 파일명 순서와 캡션 순서가 일치)**
```
A child in a pink dress is climbing up stairs
A dog playing in the park
```

### 3. GloVe 임베딩 다운로드 (선택사항)

사전 학습된 워드 임베딩을 사용하면 성능이 향상됩니다.

```bash
# 다운로드
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip

# assets 폴더로 이동
mv glove.6B.300d.txt assets/
```

GloVe 파일이 없어도 랜덤 초기화된 임베딩으로 학습이 가능합니다.

## 🎓 학습 방법

### 기본 학습

```bash
python scripts/train.py
```

### 학습 과정

1. **단어장 생성**: 캡션 파일에서 단어를 추출하여 단어장 생성
2. **GloVe 로드**: 사전 학습된 임베딩 로드 (있는 경우)
3. **데이터셋 로드**: 이미지와 캡션 매칭
4. **모델 초기화**: MobileNet 인코더 + GRU 디코더
5. **학습 실행**: 각 epoch마다:
   - 학습 진행
   - 여러 샘플로 캡션 생성 및 검증
   - 모델 저장 (5 epoch마다)

### 출력 예시

```
단어장 생성 중...
단어장 크기: 1234
GloVe 임베딩 로드 중: assets/glove.6B.300d.txt
✅ GloVe 임베딩 로드 완료: 400000개 단어
✅ 임베딩 행렬 생성 완료: 800/1230개 단어 매칭
데이터셋 로드 중...
로드된 데이터: 40456개의 이미지-캡션 쌍
학습 시작 (Encoder Frozen)... 총 40456개 샘플, 10 에포크

Epoch [1/10], Step [0/633], Loss: 8.6084
...
=== Epoch 1/10 완료. 평균 Loss: 4.0171 ===

🔍 검증: 5개 샘플로 캡션 생성 및 평가
[샘플 1/5]
  📸 이미지: 1000268201_693b08cb0e.jpg
  📝 원본: A child in a pink dress is climbing up stairs
  🤖 생성: a child in a pink dress climbing stairs
  📊 일치율: 75.0% (6/8 단어)
...
```

## 💻 사용 방법

### 학습된 모델로 캡션 생성

```python
import torch
from src.muti_modal_model.model import MobileNetCaptioningModel
from PIL import Image
from torchvision import transforms

# 모델 로드
checkpoint = torch.load('lightweight_captioning_model.pth')
word_map = checkpoint['word_map']
rev_word_map = checkpoint['rev_word_map']

model = MobileNetCaptioningModel(vocab_size=checkpoint['vocab_size'], embed_dim=300)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

image = Image.open('test_image.jpg').convert('RGB')
image_tensor = transform(image).unsqueeze(0)

# 캡션 생성
with torch.no_grad():
    caption = model.generate(image_tensor, word_map, rev_word_map, max_len=50)
    print(' '.join([w for w in caption if w not in ['<start>', '<end>', '<pad>', '<unk>']]))
```

## 🤖 Jetson Nano 설정

### 1. 시스템 요구사항

- Jetson Nano (4GB 또는 8GB)
- JetPack 4.6 이상
- Python 3.6 이상

### 2. JetPack 설치

[Jetson Nano 개발자 키트 설정 가이드](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit)를 참고하여 JetPack을 설치합니다.

### 3. PyTorch 설치

Jetson Nano용 PyTorch는 사전 빌드된 wheel 파일을 사용합니다.

```bash
# PyTorch 1.12.0 (JetPack 4.6용)
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.12.0-cp38-cp38-linux_aarch64.whl
pip3 install torch-1.12.0-cp38-cp38-linux_aarch64.whl

# torchvision 설치
sudo apt-get install libopenblas-base libopenmpi-dev libomp-dev
pip3 install torchvision
```

### 4. 추가 패키지 설치

```bash
# 시스템 패키지
sudo apt-get update
sudo apt-get install python3-pip python3-dev

# Python 패키지
pip3 install -r requirements.txt
```

### 5. Jetson Nano 최적화 설정

```bash
# 전력 모드 설정 (최대 성능)
sudo nvpmodel -m 0
sudo jetson_clocks

# 스왑 메모리 증가 (필요시)
sudo systemctl disable nvzramconfig
sudo fallocate -l 4G /mnt/4GB.swap
sudo chmod 600 /mnt/4GB.swap
sudo mkswap /mnt/4GB.swap
sudo swapon /mnt/4GB.swap
```

### 6. 학습 실행

```bash
# 배치 크기 조정 (메모리에 따라)
# scripts/train.py에서 BATCH_SIZE를 16 또는 8로 설정

python3 scripts/train.py
```

### 7. Jetson Nano 특화 설정

`scripts/train.py`에서 다음 설정을 조정하세요:

```python
# Jetson Nano용 최적화
BATCH_SIZE = 8  # 메모리에 따라 조정
NUM_WORKERS = 0  # Jetson에서는 0 권장
USE_MIXED_PRECISION = False  # Jetson에서는 FP32 권장
```

## ☁️ Google Colab 사용

### 1. 저장소 클론 및 설정

```python
# Colab 노트북에서
from google.colab import drive
drive.mount('/content/drive')

!git clone <your-repo-url>
%cd 임베디드

# 패키지 설치
!pip install torch torchvision pillow
```

### 2. 데이터 준비

```python
# 방법 1: Google Drive에 데이터 업로드
# /content/drive/MyDrive/assets/images/
# /content/drive/MyDrive/assets/captions.txt

# 방법 2: 직접 업로드
from google.colab import files
# files.upload()  # 파일 업로드
```

### 3. GloVe 다운로드 (선택사항)

```python
!wget http://nlp.stanford.edu/data/glove.6B.zip
!unzip glove.6B.zip
!mv glove.6B.300d.txt /content/drive/MyDrive/assets/
```

### 4. 학습 실행

```python
!python scripts/train.py
```

Colab 환경에서는 자동으로 다음 경로를 사용합니다:
- 이미지: `/content/drive/MyDrive/assets/images/`
- 캡션: `/content/drive/MyDrive/assets/captions.txt`
- 모델 저장: `/content/drive/MyDrive/models/`

## ⚙️ 설정 옵션

`scripts/train.py` 파일에서 다음 설정을 변경할 수 있습니다:

```python
# 학습 설정
LEARNING_RATE = 4e-4      # 학습률
BATCH_SIZE = 64           # 배치 크기 (GPU), 16 (CPU)
EPOCHS = 10               # 에포크 수
MAX_CAPTION_LEN = 50      # 최대 캡션 길이
MIN_WORD_FREQ = 2         # 단어장 최소 빈도

# 모델 설정
ENCODER_FINE_TUNING = True  # Epoch 5 이후 인코더 미세조정
USE_MIXED_PRECISION = True  # FP16 학습 (CUDA/MPS)
EMBED_DIM = 300            # 임베딩 차원 (GloVe 사용 시 300)

# 검증 설정
VAL_NUM_SAMPLES = 5        # 검증 샘플 수
```

## 📊 성능 최적화

### GPU 환경별 예상 성능

| 환경 | 배치 크기 | 1 Epoch 시간 | 10 Epoch 시간 |
|------|----------|-------------|--------------|
| Mac MPS (M1/M2) | 64 | ~5-10분 | ~50-100분 |
| Colab T4 GPU | 128 | ~2-4분 | ~20-40분 |
| Colab V100 GPU | 256 | ~1-2분 | ~10-20분 |
| Jetson Nano | 8 | ~30-60분 | ~5-10시간 |

### 최적화 팁

1. **배치 크기 조정**: GPU 메모리에 맞게 조정
2. **Mixed Precision**: CUDA/MPS에서 활성화 시 속도 향상
3. **GloVe 사용**: 사전 학습된 임베딩으로 성능 향상
4. **데이터 증강**: 이미지 데이터 증강으로 성능 향상 가능

## 🐛 문제 해결

### 메모리 부족 오류

```python
# 배치 크기 감소
BATCH_SIZE = 16  # 또는 8
```

### GloVe 파일을 찾을 수 없음

- 파일 경로 확인: `assets/glove.6B.300d.txt`
- 파일이 없어도 랜덤 초기화로 학습 가능

### CUDA out of memory

```python
# 배치 크기 감소 또는 Mixed Precision 비활성화
BATCH_SIZE = 32
USE_MIXED_PRECISION = False
```

## 📝 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.

## 🙏 참고 자료

- [MobileNetV3 Paper](https://arxiv.org/abs/1905.02244)
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
- [Jetson Nano Developer Kit](https://developer.nvidia.com/embedded/jetson-nano-developer-kit)

## 📧 문의

문제가 발생하거나 질문이 있으시면 이슈를 등록해주세요.

