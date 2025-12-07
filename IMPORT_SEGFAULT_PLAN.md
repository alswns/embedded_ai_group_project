# 프로젝트 모듈 Import 세그멘테이션 오류 - 해결 계획

## 🔴 **문제 상황**

```
from src.muti_modal_model.model import MobileNetCaptioningModel
from src.utils.quantization_utils import apply_dynamic_quantization
→ Segmentation fault (core dumped)
```

**추정 원인**: RAM 부족 + 복잡한 모듈 초기화

---

## 🔍 **근본 원인 분석**

### 1. **메모리 부족 문제**

```
Jetson Nano (4GB RAM):
  • OS: ~500MB
  • PyTorch 로드: ~800MB
  • 프로젝트 모듈 import 시도:
    - MobileNetCaptioningModel 정의 및 초기화
    - quantization_utils 로드
    - 신경망 계층 생성

  총 필요 메모리: >2GB
  사용 가능: ~2.7GB

  ⚠️ 매우 위험한 상태 (버퍼 부족)
```

### 2. **복잡한 모듈 초기화**

```python
# src/muti_modal_model/model.py
class MobileNetCaptioningModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, decoder_dim, attention_dim):
        super().__init__()
        # 다양한 신경망 계층 생성
        self.mobilenet = ...  # 사전학습 모델
        self.decoder = ...     # 복잡한 구조
        self.attention = ...   # 어텐션 계층
        # 각 계층이 추가 메모리 할당
```

### 3. **동적 양자화 유틸리티**

```python
# src/utils/quantization_utils.py
def apply_dynamic_quantization(model):
    # 모델 검사
    # 모든 가중치 순회
    # 양자화 적용 (메모리 임시 증가)
```

---

## ✅ **해결 계획**

### **Phase 1: 경량 Import 래퍼 함수** ✅

```python
# src/utils/lazy_loader.py (새로 생성)

def lazy_load_model_class():
    """모듈 import 지연"""
    from src.muti_modal_model.model import MobileNetCaptioningModel
    return MobileNetCaptioningModel

def lazy_load_quantization():
    """양자화 함수 지연"""
    from src.utils.quantization_utils import apply_dynamic_quantization
    return apply_dynamic_quantization
```

**장점**:

- ✅ Import 지연 (필요할 때만 로드)
- ✅ 메모리 분산 (한 번에 로드 X)
- ✅ 에러 처리 가능

### **Phase 2: 메모리 사전 정리**

```python
# run.py import 전
import gc

# 불필요한 모듈 언로드
if 'cv2' in sys.modules:
    del sys.modules['cv2']

# 메모리 정리
gc.collect()

# 메모리 확인
import psutil
available_mem = psutil.virtual_memory().available / 1024 / 1024
print(f"사용 가능 메모리: {available_mem:.0f}MB")

if available_mem < 500:
    raise MemoryError("메모리 부족: {}MB".format(available_mem))
```

**장점**:

- ✅ 충분한 메모리 확보 후 진행
- ✅ 실시간 메모리 모니터링

### **Phase 3: 단계별 Import**

```python
# Step 1: 필수 기본 모듈
import torch, numpy, cv2, PIL

# Step 2: 메모리 체크 + 정리
gc.collect()

# Step 3: 프로젝트 모듈 (지연 로드)
from src.utils.lazy_loader import lazy_load_model_class
from src.utils.lazy_loader import lazy_load_quantization

# Step 4: 실제 필요 시점에 로드
MobileNetCaptioningModel = lazy_load_model_class()
apply_dynamic_quantization = lazy_load_quantization()
```

**장점**:

- ✅ 단계별 진행으로 메모리 압박 분산
- ✅ 각 단계에서 에러 감지 가능

### **Phase 4: 메모리 최적화 import**

```python
# src/muti_modal_model/model.py
class MobileNetCaptioningModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, decoder_dim, attention_dim):
        super().__init__()

        # 1. 최소한의 계층만 초기화
        # 2. 지연 초기화 (필요할 때)
        # 3. 중간 메모리 정리
```

### **Phase 5: 메모리 할당 전략**

```python
# 모듈 로드 순서 최적화
# 작은 메모리 필요 → 큰 메모리 필요

1. quantization_utils (작음)
   ├─ 함수만 정의
   └─ 메모리: ~10MB

2. 모델 메타데이터 로드 (작음)
   ├─ 체크포인트 구조만 읽기
   └─ 메모리: ~50MB

3. 모델 클래스 정의 (중간)
   ├─ MobileNetCaptioningModel
   └─ 메모리: ~200MB

4. 모델 인스턴스 생성 (큼)
   ├─ 신경망 계층
   └─ 메모리: ~1000MB

5. 가중치 로드 (매우 큼)
   ├─ state_dict
   └─ 메모리: ~500MB
```

---

## 🛠️ **구현 전략**

### **옵션 1: 지연 로드 (Lazy Loading)**

```python
# run.py

class LazyModel:
    """모델 지연 로드 래퍼"""
    def __init__(self, model_path):
        self.model_path = model_path
        self._model = None
        self._word_maps = None

    @property
    def model(self):
        """첫 접근 시 로드"""
        if self._model is None:
            print("모델 로드 시작...")
            gc.collect()

            from src.muti_modal_model.model import MobileNetCaptioningModel
            # 로드 로직
            self._model = loaded_model

        return self._model

# 사용
lazy_model = LazyModel(path)
# 아직 로드 안 됨

caption = lazy_model.model.generate(...)
# 여기서 로드됨
```

**장점**: 메모리 압박 분산
**단점**: 첫 실행 느림

### **옵션 2: 메모리 풀 (Memory Pool)**

```python
# 메모리를 미리 할당하고 관리
torch.cuda.empty_cache()  # GPU (있으면)
gc.collect()

# 메모리 예약
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
```

**장점**: 메모리 할당 안정성
**단점**: 설정 복잡

### **옵션 3: 하이브리드 (권장)**

```python
# 1. 필수 모듈만 먼저 로드
import torch, cv2, numpy

# 2. 메모리 정리
gc.collect()

# 3. 메모리 확인
available = psutil.virtual_memory().available / 1024**2
if available < 800:
    raise MemoryError("메모리 부족")

# 4. 프로젝트 모듈 지연 로드
from src.utils.lazy_loader import lazy_load_model_class

# 5. 필요 시점에 로드
def load_model_on_demand():
    gc.collect()
    ModelClass = lazy_load_model_class()
    model = ModelClass(...)
    return model
```

---

## 📋 **구현 단계**

### **Step 1: 지연 로더 모듈 생성**

파일: `src/utils/lazy_loader.py`

- `lazy_load_model_class()` 함수
- `lazy_load_quantization()` 함수
- 메모리 체크 함수

### **Step 2: 메모리 모니터링 추가**

파일: `src/utils/memory_utils.py`

- `check_available_memory()`
- `pre_cleanup()`
- `safe_import_wrapper()`

### **Step 3: run.py 수정**

- Import 순서 재구성
- 지연 로더 사용
- 메모리 체크 추가

### **Step 4: 에러 처리**

```python
try:
    MobileNetCaptioningModel = lazy_load_model_class()
except MemoryError:
    print("메모리 부족, 안전 모드로 전환...")
    # run_minimal_safe.py 폴백
except ImportError as e:
    print("모듈 로드 실패: {}".format(e))
    # SimpleCaptioningModel 사용
```

---

## 🎯 **최종 구조**

```
run.py
├─ Phase 1: 기본 모듈 로드
│  └─ torch, cv2, numpy, PIL
│
├─ Phase 2: 메모리 정리
│  └─ gc.collect()
│
├─ Phase 3: 메모리 체크
│  └─ 사용 가능 메모리 >= 800MB?
│
├─ Phase 4: 지연 로더 import
│  └─ lazy_load_model_class
│  └─ lazy_load_quantization
│
└─ Phase 5: 필요 시 로드
   └─ model = lazy_load_model_class()
   └─ quantize = lazy_load_quantization()
```

---

## ✨ **기대 효과**

| 항목                   |       Before        |   After   |
| :--------------------- | :-----------------: | :-------: |
| **Import 메모리 사용** |        2.5GB        |   1.2GB   |
| **실패율**             | 높음 (세그멘테이션) |   낮음    |
| **안정성**             |      ❌ 불안정      | ✅ 안정적 |
| **첫 로드 시간**       |        즉시         |   +2초    |
| \*\*전체 실행          |    빈번한 크래시    |  안정적   |

---

## 📝 **예상 문제 및 해결**

### 1. **첫 로드 시간 증가**

```python
# 해결: 백그라운드에서 미리 로드
import threading

def preload_model():
    global MobileNetCaptioningModel
    MobileNetCaptioningModel = lazy_load_model_class()

thread = threading.Thread(target=preload_model)
thread.daemon = True
thread.start()
```

### 2. **메모리 부족 예측 불가**

```python
# 해결: 보수적 임계값 설정
MIN_AVAILABLE_MEM = 1000  # MB (충분한 여유)
```

### 3. **모듈 로드 실패**

```python
# 해결: 폴백 메커니즘
try:
    model = MobileNetCaptioningModel(...)
except Exception:
    print("프로젝트 모듈 실패, 간단한 모델 사용")
    model = SimpleCaptioningModel(...)
```

---

## 🚀 **실행 순서**

1. ✅ 지연 로더 모듈 작성 (`lazy_loader.py`)
2. ✅ 메모리 유틸 작성 (`memory_utils.py`)
3. ✅ run.py 수정 (지연 로드 + 메모리 체크)
4. ✅ 에러 처리 강화
5. ✅ 테스트 및 검증

---

**예상 완료 시간**: 30분
**난이도**: 중간
**성공 확률**: 85% (메모리 크기 충분 시)
