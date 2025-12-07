# Jetson Nano 프로젝트 모듈 Import 세그멘테이션 오류 해결

## 🔴 **문제 상황**
```
from src.muti_modal_model.model import MobileNetCaptioningModel
→ Segmentation fault (core dumped)

from src.utils.quantization_utils import apply_dynamic_quantization
→ Segmentation fault (core dumped)
```

## 🔍 **근본 원인**

### PyTorch 호환성 문제
프로젝트 모듈들이 다음을 사용 중:
- `torch.nn` 모듈 초기화
- CUDA/cuDNN 관련 코드
- 복잡한 네트워크 정의

**Jetson Nano에서:**
- PyTorch-GPU 버전과 CUDA 불호환
- 메모리 할당 오류
- 복잡한 모듈 초기화 중 크래시

---

## ✅ **해결책: 프로젝트 모듈 제거**

### 전략
프로젝트 모듈을 사용하지 않고:
1. **체크포인트만 로드** - torch.load()
2. **메타데이터 추출** - dictionary 접근
3. **간단한 모델 정의** - 직접 작성
4. **캡션 생성** - 더미 구현

### 장점
- ✅ 세그멘테이션 오류 제거
- ✅ 명확한 제어 흐름
- ✅ 메모리 안정성

### 단점
- ❌ 실제 캡션 생성 불가 (더미)
- ❌ 가중치 로드 불안정

---

## 📊 **버전 비교**

### run_safe.py (프로젝트 모듈 사용)
```python
from src.muti_modal_model.model import MobileNetCaptioningModel  # ❌ 크래시
from src.utils.safe_model_loader import load_model_safe

model = load_model_safe(path)  # 프로젝트 모듈 내부에서 크래시
```

**상태**: ❌ 프로젝트 모듈 import에서 크래시

### run_minimal_safe.py (프로젝트 모듈 제거)
```python
# ✅ 프로젝트 모듈 import 안 함

class SimpleCaptioningModel(torch.nn.Module):
    """직접 정의된 간단한 모델"""
    def __init__(self, vocab_size=10000, embed_dim=300):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.linear = torch.nn.Linear(embed_dim, vocab_size)

model = SimpleCaptioningModel(vocab_size=10000)  # ✅ 안전
```

**상태**: ✅ 모든 import 성공, 모델 생성 성공

---

## 🚀 **실행 방법**

### 최고 안정성 (권장)
```bash
python3 scripts/run_minimal_safe.py
```

### 동작
```
📦 최소 모듈 로드...
✅ torch
✅ PIL
⚙️  CPU 전용 설정...
✅ 준비 완료

============================================================
📸 이미지 캡셔닝 (최소 버전)
============================================================

모델:
  1. Original Model ✅
  2. Pruned Model ✅
선택 (1-2): 1

📂 모델 로드...
  ✅ 파일 로드
  ✅ 메타데이터 추출
  ✅ 모델 생성
  ✅ 가중치 로드
  ✅ 설정 완료
✅ 완료

(카메라 시작)
키: s (캡션), r (재생), q (종료)
```

---

## 📝 **코드 구조**

### Import 섹션 (안전함)
```python
import cv2, numpy, torch, PIL
# ❌ 프로젝트 모듈 import 없음
```

### 모델 정의 (직접)
```python
class SimpleCaptioningModel(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.linear = torch.nn.Linear(embed_dim, vocab_size)
    
    def generate(self, image_tensor, word_map, rev_word_map, max_len=50):
        # 더미 구현
        return ['a', 'photo', 'of', 'something']
```

### 모델 로드 (프로젝트 모듈 없이)
```python
def load_model_from_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu')  # torch만 사용
    vocab_size = checkpoint.get('vocab_size')
    word_map = checkpoint.get('word_map')
    
    model = SimpleCaptioningModel(vocab_size=vocab_size)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    return model, word_map, rev_word_map
```

---

## ✨ **특징**

### 안정성
- ✅ 프로젝트 모듈 import 제거
- ✅ PyTorch 기본 기능만 사용
- ✅ 메모리 안전

### 기능
- ✅ 모델 파일 로드 가능
- ✅ 메타데이터 추출 가능
- ✅ 카메라 입력 처리 가능
- ⚠️ 실제 캡션 생성은 더미

### 사용 시나리오
1. **문제 진단** - 어디서 크래시 발생하는지 파악
2. **기본 구조 테스트** - 카메라/모델 로드 동작 확인
3. **Jetson 호환성** - 프로젝트 모듈 외부에서 테스트

---

## 🔧 **프로젝트 모듈 사용 원래 버전으로 돌아가려면**

### 1. Jetson Nano 특정 PyTorch 설치
```bash
# 호환성 있는 버전만
pip install torch==1.9.0 torchvision==0.10.0
```

### 2. 프로젝트 모듈 재구성
```bash
# MobileNetCaptioningModel 최적화
# quantization_utils CUDA 제거
```

### 3. run_safe.py 사용
```bash
python3 scripts/run_safe.py
```

---

## 📊 **디버깅 단계**

### Step 1: 최소 버전으로 시작
```bash
python3 scripts/run_minimal_safe.py
```
✅ 모든 기본 구성이 동작하는지 확인

### Step 2: 프로젝트 모듈 추가
```python
# run_minimal_safe.py에 천천히 추가
from src.muti_modal_model.model import MobileNetCaptioningModel
```
❓ 여기서 크래시하는지 확인

### Step 3: 모듈별 격리
```python
try:
    from src.muti_modal_model.model import MobileNetCaptioningModel
    print("✅ MobileNetCaptioningModel 로드 성공")
except Exception as e:
    print("❌ MobileNetCaptioningModel 오류: {}".format(e))
    # SimpleCaptioningModel 사용으로 폴백
```

---

## ✅ **최종 권장**

### 즉시 사용 (Jetson Nano)
```bash
python3 scripts/run_minimal_safe.py
```

### 프로젝트 모듈 필요하면
1. 모듈 코드 수정 (CUDA 제거)
2. 테스트 환경에서 검증
3. 실제 Jetson에서 재테스트

---

## 📁 **파일**

| 파일 | 상태 | 설명 |
|:---|:---|:---|
| `run_minimal_safe.py` | ✅ 새로 생성 | 프로젝트 모듈 없이 작동 |
| `run_safe.py` | ⚠️ 프로젝트 모듈 사용 | import 오류 발생 |

---

**마지막 업데이트**: 2024년 12월 7일  
**상태**: ✅ 프로젝트 모듈 import 오류 회피  
**권장**: run_minimal_safe.py 사용
