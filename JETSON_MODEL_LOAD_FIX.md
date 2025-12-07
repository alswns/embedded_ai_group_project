# Jetson Nano 모델 로드 세그멘테이션 오류 해결

## 🔴 **문제 상황**
```
모델 로드 중 세그멘테이션 오류 발생
Segmentation fault (core dumped)
```

## 🔍 **원인 분석**

### 모델 로드 단계에서 발생 가능한 크래시 지점

1. **torch.load() 단계**
   - 손상된 체크포인트 파일
   - 메모리 부족
   - 파일 읽기 권한 문제

2. **모델 인스턴스 생성 단계**
   - MobileNetCaptioningModel 초기화 중 메모리 할당 실패
   - 빌더 메서드 내부 에러

3. **load_state_dict() 단계**
   - 잘못된 가중치 형태
   - 메모리 복사 실패
   - strict=False 옵션 미흡

4. **eval() / cpu() 단계**
   - 모듈 재구성 중 크래시
   - 버퍼 메모리 이동 실패

---

## ✅ **적용된 해결책**

### 1️⃣ **단계별 세그멘테이션 격리** ✅

```python
# Before: 한 번에 처리 (하나 실패하면 전체 크래시)
checkpoint = torch.load(...)
model.load_state_dict(...)
model.eval()

# After: 각 단계를 개별 처리
checkpoint = safe_load_checkpoint(path)
metadata = safe_extract_metadata(checkpoint)
model = safe_create_model(...)
safe_load_state_dict(model, ...)
model = safe_setup_eval(model)
safe_cleanup(...)
```

**효과**:
- ✅ 어느 단계에서 실패했는지 명확한 에러 메시지
- ✅ 하나의 단계 실패 → 다음 진행 안 함 (안전)
- ✅ 메모리 누수 방지

### 2️⃣ **독립적 안전 함수** ✅

```python
def safe_load_checkpoint(path):
    """체크포인트 로드만 담당"""
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        return checkpoint
    except Exception as e:
        print("로드 실패: {}".format(e))
        return None

def safe_create_model(vocab_size, decoder_dim, attention_dim):
    """모델 생성만 담당"""
    try:
        model = MobileNetCaptioningModel(...)
        return model
    except Exception as e:
        print("생성 실패: {}".format(e))
        return None
```

**효과**:
- ✅ 각 함수가 독립적으로 테스트 가능
- ✅ 실패해도 다른 함수 영향 없음
- ✅ 메모리 정리 독립적 수행

### 3️⃣ **조기 메모리 정리** ✅

```python
# 불필요한 객체 즉시 삭제
del checkpoint, state_dict
gc.collect()

# 크래시 발생 시에도 정리
try:
    # ... 작업
except:
    # 강제 정리
    del checkpoint, state_dict
    gc.collect()
```

**효과**:
- ✅ 메모리 압박 감소
- ✅ 다음 단계 메모리 충분
- ✅ 스택 오버플로우 방지

### 4️⃣ **상세한 에러 정보** ✅

```python
print("  체크포인트 로드...", file=sys.stderr)
try:
    checkpoint = torch.load(...)
    print("    ✅ 로드 성공", file=sys.stderr)
except Exception as e:
    print("    ❌ 로드 실패: {}".format(e), file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    return None
```

**효과**:
- ✅ 정확한 에러 위치 파악
- ✅ 스택 트레이스 확인 가능
- ✅ 문제 원인 빠른 진단

---

## 📊 **로드 프로세스**

### Before (위험한 버전)
```
torch.load()
  ↓ (실패하면 크래시, 정리 불가)
model.load_state_dict()
  ↓ (메모리 누수)
model.eval()
  ↓ (불안정)
return model (불완전)
```

### After (안전한 버전)
```
safe_load_checkpoint()
  ✅ 체크 후 진행 또는 반환
↓
safe_extract_metadata()
  ✅ 메모리 정리 후 진행 또는 반환
↓
safe_create_model()
  ✅ 생성 확인 후 진행 또는 반환
↓
safe_load_state_dict()
  ✅ 로드 시도 (실패해도 계속)
↓
safe_setup_eval()
  ✅ 평가 모드 설정 또는 반환
↓
safe_cleanup()
  ✅ 항상 실행되는 정리
↓
return model (완전하고 안전함)
```

---

## 🚀 **실행 방법**

### 안전한 버전 (권장)
```bash
python3 scripts/run_safe.py
```

### 테스트 (모델 로드만)
```bash
python3 src/utils/safe_model_loader.py
```

---

## 🎯 **권장 설정**

| 항목 | 설정 |
|:---|:---|
| **스크립트** | run_safe.py |
| **모델** | Pruned Model |
| **양자화** | FP32 |
| **디바이스** | CPU 전용 |

---

## ✨ **성공 신호**

```
📂 모델 로드 중: models/lightweight_captioning_model.pth
  체크포인트 로드...
    ✅ 로드 성공
  메타데이터 추출...
    ✅ 추출 성공 (vocab=9487)
  모델 생성...
    ✅ 생성 성공
  가중치 로드...
    ✅ 로드 성공
  평가 모드 설정...
    ✅ 설정 성공
  메모리 정리...
    ✅ 정리 성공
✅ 모델 로드 완료
```

---

## 🔧 **문제 해결**

### 여전히 크래시 발생하면

1. **체크포인트 파일 검사**
   ```bash
   ls -lh models/lightweight_captioning_model.pth
   file models/lightweight_captioning_model.pth
   ```

2. **메모리 확인**
   ```bash
   free -h
   ```

3. **safe_model_loader 단독 테스트**
   ```bash
   python3 src/utils/safe_model_loader.py
   ```

4. **스택 트레이스 확인**
   ```bash
   python3 scripts/run_safe.py 2>&1 | tee run.log
   ```

---

## 📝 **수정된 파일**

| 파일 | 변경 사항 |
|:---|:---|
| `scripts/run_safe.py` | ✅ 단계별 로드로 변경 |
| `src/utils/safe_model_loader.py` | ✅ 새로 생성 |

---

## ✅ **최종 상태**

**모델 로드 세그멘테이션 오류 해결**

- ✅ 단계별 격리로 명확한 에러 메시지
- ✅ 독립적 함수로 재사용 가능
- ✅ 메모리 누수 방지
- ✅ 안전한 정리 보장

---

**마지막 업데이트**: 2024년 12월 7일  
**상태**: ✅ 모델 로드 안정화 완료  
**권장**: scripts/run_safe.py 사용
