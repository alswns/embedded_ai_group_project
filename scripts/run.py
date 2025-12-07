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

print("📦 모듈 로드 시작...", file=sys.stderr)

try:
    from PIL import Image
    print("   ✅ PIL 로드", file=sys.stderr)
except ImportError as e:
    print("❌ PIL 필요: {}".format(e), file=sys.stderr)
    sys.exit(1)

try:
    from gtts import gTTS
    print("   ✅ gtts 로드", file=sys.stderr)
except ImportError:
    print("   ⚠️  gtts 미사용", file=sys.stderr)

try:
    import pygame
    print("   ✅ pygame 로드", file=sys.stderr)
except ImportError:
    print("   ⚠️  pygame 미사용", file=sys.stderr)

# 프로젝트 모듈 지연 로드
print("   ℹ️  프로젝트 모듈 (지연 로드 준비)", file=sys.stderr)

# 지연 로더 import (매우 간단함)
from src.utils.memory_safe_import import load_model_class
print("   ✅ 지연 로더 로드", file=sys.stderr)

# 아직 실제 로드는 안 됨
_model_class_loader = load_model_class
    


print("✅ 모든 모듈 로드 완료", file=sys.stderr)

# ============================================================================
# 환경 설정 (CRITICAL - 크래시 방지)
# ============================================================================
print("⚙️  환경 설정 중...", file=sys.stderr)
torch.backends.cudnn.enabled = False  # 불안정성 방지
torch.backends.cudnn.benchmark = True # 입력 크기가 고정(224x224)이므로 필수

# CPU/GPU 디바이스 자동 감지 및 강제 설정
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("🚀 디바이스: GPU (NVIDIA Maxwell) 가속 모드", file=sys.stderr)
else:
    device = torch.device("cpu")
    print("📍 디바이스: CPU (경고: 성능이 낮을 수 있음)", file=sys.stderr)

# 스레드 최적화
torch.set_num_threads(4)
torch.set_num_interop_threads(4)

sys.modules['numpy._core'] = np.core
sys.modules['numpy._core.multiarray'] = np.core.multiarray
dtypes = torch.float32
# ============================================================================
# 이미지 전처리 함수 (torchvision 대체)
# ============================================================================
def preprocess_image_optimized(frame):
    """
    Jetson Nano 최적화 전처리:
    1. PIL 제거 (느림) -> OpenCV 사용 (빠름)
    2. CPU 연산 최소화 -> GPU로 바로 업로드
    """
    # 1. OpenCV 리사이즈 (CPU 부하 감소)
    img = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
    
    # 2. BGR -> RGB 및 정규화 (Numpy)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    # 3. 정규화 (Mean/Std)
    img -= np.array([0.485, 0.456, 0.406], dtype=np.float32)
    img /= np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    # 4. (H, W, C) -> (C, H, W)
    img = np.transpose(img, (2, 0, 1))
    
    # 5. Tensor 변환 및 GPU 업로드
    image_tensor = torch.from_numpy(img).unsqueeze(0).to(device)
    
    # ★ 핵심: GPU 모드일 경우 Half Precision(FP16) 적용
    return image_tensor.float()

preprocess_image = preprocess_image_optimized

# 모델 경로 설정
MODELS = {
    '1': {
        'name': 'Original Model',
        'path': 'models/lightweight_captioning_model.pth',
        'fallback': 'lightweight_captioning_model.pth'
    },
    '2': {
        'name': 'Pruned Model (Struct 30% + Mag 10%)',
        'path': 'pruning_results/Pruning_epoch_1_checkpoint.pt',
        'fallback': None
    }
}

# 양자화 옵션
QUANTIZE_OPTIONS = {
    '1': {'name': 'FP32 (원본)', 'enabled': False},
    '2': {'name': 'FP16 (Half Precision)', 'enabled': True},
}

print("✅ 환경 설정 완료", file=sys.stderr)
# ============================================================================
# 성능 모니터링 클래스
# ============================================================================
class PerformanceMonitor:
    """모델 성능 모니터링"""
    def __init__(self,model):
        self.inference_times = []
        self.memory_usage = []
        self.gpu_memory = []
        self.process = psutil.Process(os.getpid())
        print("모델 크기 : {:.2f} MB".format(sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024))
    def record_inference(self, inference_time):
        """추론 시간 기록"""
        self.inference_times.append(inference_time)
    
    def get_cpu_memory_mb(self):
        """CPU 메모리 사용량 (MB)"""
        try:
            mem_info = self.process.memory_info()
            return mem_info.rss / 1024 / 1024
        except:
            return 0.0
    
    def get_gpu_memory_mb(self):
        """GPU 메모리 사용량 (MB)"""
        if device.type == 'cuda':
            return torch.cuda.memory_allocated() / 1024 / 1024
        elif device.type == 'mps':
            try:
                return torch.mps.current_allocated_memory() / 1024 / 1024
            except:
                return 0.0
        return 0.0
    
    def record_memory(self):
        """메모리 사용량 기록"""
        self.memory_usage.append(self.get_cpu_memory_mb())
        self.gpu_memory.append(self.get_gpu_memory_mb())
    
    def get_stats(self):
        """통계 계산"""
        if not self.inference_times:
            return None
        
        inf_times = np.array(self.inference_times[-30:])  # 최근 30개
        
        stats = {
            'mean_latency_ms': float(np.mean(inf_times)),
            'median_latency_ms': float(np.median(inf_times)),
            'min_latency_ms': float(np.min(inf_times)),
            'max_latency_ms': float(np.max(inf_times)),
            'std_latency_ms': float(np.std(inf_times)),
            'fps': float(1000.0 / np.mean(inf_times)),
            'cpu_memory_mb': float(np.mean(self.memory_usage[-30:]) if self.memory_usage else 0),
            'gpu_memory_mb': float(np.mean(self.gpu_memory[-30:]) if self.gpu_memory else 0),
            'total_inferences': len(self.inference_times)
        }
        return stats
    
    def print_stats(self):
        """성능 통계 출력"""
        stats = self.get_stats()
        if stats is None:
            print("아직 데이터가 없습니다.")
            return
        
        print("\n" + "="*70)
        print("=== 성능 통계 (JTOPS 스타일) ===")
        print("="*70)
        print("⏱️  추론 시간 (Latency):")
        print("    • 평균: {:.2f} ms".format(stats['mean_latency_ms']))
        print("    • 중앙값: {:.2f} ms".format(stats['median_latency_ms']))
        print("    • 최소/최대: {:.2f} / {:.2f} ms".format(stats['min_latency_ms'], stats['max_latency_ms']))
        print("    • 표준편차: {:.2f} ms".format(stats['std_latency_ms']))
        print("\n🎬 처리 속도 (Throughput):")
        print("    • FPS: {:.1f} frame/sec".format(stats['fps']))
        print("    • 1프레임 처리: {:.2f} ms".format(stats['mean_latency_ms']))
        print("\n💾 메모리 사용량:")
        print("    • CPU: {:.1f} MB".format(stats['cpu_memory_mb']))
        if device.type in ['cuda', 'mps']:
            print("    • GPU: {:.1f} MB".format(stats['gpu_memory_mb']))
        print("\n📊 누적 통계:")
        print("    • 총 추론 횟수: {}회".format(stats['total_inferences']))
        print("="*70 + "\n")

# ============================================================================
# 모델 선택 함수
# ============================================================================
def select_model():
    """사용할 모델 선택"""
    print("\n" + "="*70)
    print("=== 사용할 모델 선택 ===")
    print("="*70)
    
    for key, model_info in MODELS.items():
        path = model_info['path']
        exists = os.path.exists(path)
        status = "✅ 사용 가능" if exists else "❌ 없음"
        print("{}. {} {}".format(key, model_info['name'], status))
    
    print()
    while True:
        choice = input("모델을 선택하세요 (1-2): ").strip()
        if choice in MODELS:
            return choice
        print("❌ 잘못된 입력입니다. 다시 선택해주세요.")

# ============================================================================
# 양자화 선택 함수
# ============================================================================
def select_quantization():
    """사용할 양자화 옵션 선택"""
    print("\n" + "="*70)
    print("=== 양자화 옵션 선택 ===")
    print("="*70)
    
    for key, quant_info in QUANTIZE_OPTIONS.items():
        enabled = "✅" if quant_info['enabled'] else "❌"
        print("{}. {} {}".format(key, quant_info['name'], enabled))
    
    print()
    while True:
        choice = input("양자화 옵션을 선택하세요 (1-3): ").strip()
        if choice in QUANTIZE_OPTIONS:
            return choice
        print("❌ 잘못된 입력입니다. 다시 선택해주세요.")

# ============================================================================
# 음성 출력 함수
# ============================================================================
def speak_text_gtts(text):
    """TTS 음성 출력"""
    def _speak():
        temp_file = None
        try:
            pygame.mixer.init()
            tts = gTTS(text=text, lang='en', slow=False)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            temp_filename = temp_file.name
            temp_file.close()
            
            tts.save(temp_filename)
            pygame.mixer.music.load(temp_filename)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
        except Exception as e:
            print("TTS Error: {}".format(e))
        finally:
            try:
                if temp_file and os.path.exists(temp_filename):
                    pygame.mixer.music.unload()
                    os.remove(temp_filename)
            except:
                pass
    
    thread = threading.Thread(target=_speak)
    thread.daemon = True
    thread.start()

# ============================================================================
# 모델 로드
# ============================================================================
def load_model(model_choice):
    """학습된 캡셔닝 모델 로드"""
    model_info = MODELS[model_choice]
    model_path = model_info['path']
    
    # 파일 존재 확인
    if not os.path.exists(model_path):
        if model_info['fallback']:
            model_path = model_info['fallback']
            if not os.path.exists(model_path):
                print("❌ 모델 파일을 찾을 수 없습니다: {}".format(model_info['path']))
                return None, None, None, None
        else:
            print("❌ 모델 파일을 찾을 수 없습니다: {}".format(model_path))
            return None, None, None, None
    
    try:
        print("\n📂 모델 로드 중: {}".format(model_path))
        
        # 프로젝트 모듈 실제 로드 (지연 로드)
        print("  1️⃣  모델 클래스 로드...", file=sys.stderr)
        try:
            Model = _model_class_loader()
            print("     ✅ 로드 완료", file=sys.stderr)
        except Exception as e:
            print("     ❌ 로드 실패: {}".format(e), file=sys.stderr)
            return None, None, None, None
        
        print("  2️⃣  체크포인트 로드...", file=sys.stderr)
        
        # CPU에서 로드 (메모리 안전) - Python/PyTorch 버전 호환성
        try:
            # Python 3.11+: weights_only 파라미터 필요
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        except TypeError:
            # Python 3.6-3.10: weights_only 파라미터 미지원
            checkpoint = torch.load(model_path, map_location=device)
        
        print("     ✅ 로드 완료", file=sys.stderr)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            word_map = checkpoint.get('word_map')
            rev_word_map = checkpoint.get('rev_word_map')
            vocab_size = checkpoint.get('vocab_size')
            
            if word_map is None or rev_word_map is None:
                print("❌ 단어장 정보가 없습니다.")
                return None, None, None, None
            
            # 체크포인트에서 모델 크기 정보 추출
            state_dict = checkpoint['model_state_dict']
            
            print("  📋 State dict 분석...", file=sys.stderr)
            print("     키 개수: {}".format(len(state_dict)), file=sys.stderr)
            
            # ★ state_dict 키 출력 (디버깅)
            for i, key in enumerate(list(state_dict.keys())[:5]):
                shape = state_dict[key].shape
                print("     [{}/{}] {}: {}".format(i+1, min(5, len(state_dict)), key, shape), file=sys.stderr)
            if len(state_dict) > 5:
                print("     ... 외 {}개".format(len(state_dict) - 5), file=sys.stderr)
            
            # ★ 핵심: state_dict에서 **실제 프루닝된 크기** 추출
            decoder_dim = checkpoint.get('decoder_dim', 512)
            attention_dim = checkpoint.get('attention_dim', 256)
            
            # state_dict에서 정확한 크기 추출 (프루닝된 실제 크기)
            if 'decoder.decode_step.weight_ih' in state_dict:
                # GRU의 input_size: (hidden_size * 3) 이므로 역으로 계산
                actual_size = state_dict['decoder.decode_step.weight_ih'].shape[0]
                actual_decoder_dim = actual_size // 3
                print("  🔍 decoder.decode_step.weight_ih 형태: {}".format(
                    state_dict['decoder.decode_step.weight_ih'].shape), file=sys.stderr)
                print("     계산된 decoder_dim: {}".format(actual_decoder_dim), file=sys.stderr)
                decoder_dim = actual_decoder_dim
            else:
                print("  ⚠️  decoder.decode_step.weight_ih 없음!", file=sys.stderr)
            
            if 'decoder.encoder_att.weight' in state_dict:
                actual_attention_dim = state_dict['decoder.encoder_att.weight'].shape[0]
                print("  🔍 decoder.encoder_att.weight 형태: {}".format(
                    state_dict['decoder.encoder_att.weight'].shape), file=sys.stderr)
                print("     계산된 attention_dim: {}".format(actual_attention_dim), file=sys.stderr)
                attention_dim = actual_attention_dim
            else:
                print("  ⚠️  decoder.encoder_att.weight 없음!", file=sys.stderr)
            
            print("   📐 최종 감지된 모델 구조 (프루닝된 크기):")
            print("      • Decoder Dim: {}".format(decoder_dim))
            print("      • Attention Dim: {}".format(attention_dim))
            print("      • Vocab Size: {}".format(vocab_size))
            
            # ★ 올바른 크기(프루닝된 크기)로 모델 생성
            print("  3️⃣  모델 인스턴스 생성 (프루닝된 크기)...", file=sys.stderr)
            try:
                # 메모리 정리
                gc.collect()
                gc.collect()
                gc.collect()
                
                # 프루닝된 크기로 모델 생성
                model = Model(
                    vocab_size=vocab_size,
                    embed_dim=300,
                    decoder_dim=decoder_dim,      # ★ 프루닝된 크기
                    attention_dim=attention_dim   # ★ 프루닝된 크기
                )
                del Model

                print("     ✅ 생성 완료 (decoder_dim={}, attention_dim={})".format(
                    decoder_dim, attention_dim), file=sys.stderr)
                
                # CPU 전환
                model = model.to(device)
                model.eval()
                
            except Exception as e:
                print("     ❌ 생성 실패: {}".format(e), file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
                return None, None, None, None
            
            # state_dict 로드 (완벽한 크기 매칭)
            print("  4️⃣  가중치 로드...", file=sys.stderr)
            try:
                # ★ 먼저 모델의 state_dict 확인
                model_state = model.state_dict()
                print("     모델 state_dict 키: {}".format(len(model_state)), file=sys.stderr)
                print("     로드할 state_dict 키: {}".format(len(state_dict)), file=sys.stderr)
                
                # ★ 누락된 키 확인
                missing_keys = set(model_state.keys()) - set(state_dict.keys())
                if missing_keys:
                    print("     ⚠️  누락된 키: {}".format(missing_keys), file=sys.stderr)
                
                unexpected_keys = set(state_dict.keys()) - set(model_state.keys())
                if unexpected_keys:
                    print("     ⚠️  예상 외 키: {}".format(unexpected_keys), file=sys.stderr)
                
                # ★ strict=True 사용: 모든 레이어가 정확히 매칭되어야 함
                model.load_state_dict(state_dict, strict=True)
                print("     ✅ 완벽한 크기 매칭으로 로드 완료", file=sys.stderr)
            except Exception as e:
                print("     ⚠️  strict=True 로드 실패: {}".format(e), file=sys.stderr)
                print("     strict=False로 재시도 중...", file=sys.stderr)
                try:
                    model.load_state_dict(state_dict, strict=False)
                    print("     ⚠️  일부 레이어만 로드됨 (프루닝 효과 감소)", file=sys.stderr)
                except Exception as e2:
                    print("     ❌ 가중치 로드 실패: {}".format(e2), file=sys.stderr)
                    print("     모델을 무작위 초기화 상태로 사용합니다.", file=sys.stderr)
                    import traceback
                    traceback.print_exc(file=sys.stderr)
            
            # 메모리 정리
            print("  5️⃣  메모리 정리...", file=sys.stderr)
            del checkpoint, state_dict
            gc.collect()
            print("     ✅ 정리 완료", file=sys.stderr)
            
            model.eval()
            
            # 모델 to CPU 명시
            try:
                model = model.to(device)
                model.eval()
            except:
                pass
            
            model_name = model_info['name']
            
            # ★ 모델 크기 정보 출력
            param_count = sum(p.numel() for p in model.parameters())
            param_size = param_count * 4 / 1024 / 1024  # FP32 기준
            
            print("\n✅ 모델 로드 완료")
            print("   모델: {}".format(model_name))
            print("   경로: {}".format(model_path))
            print("   총 파라미터: {:,}개".format(param_count))
            print("   모델 크기: {:.2f} MB (FP32)".format(param_size))
            print("   디코더 차원: {} ".format(decoder_dim))
            print("   어텐션 차원: {} ".format(attention_dim))
            
            return model, word_map, rev_word_map, model_name
        else:
            print("❌ 잘못된 모델 파일 형식입니다.")
            return None, None, None, None
            
    except Exception as e:
        print("❌ 모델 로드 실패: {}".format(e))
        import traceback
        traceback.print_exc()
        return None, None, None, None
    
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=640,
    display_height=480,
    framerate=30,
    flip_method=0,
):
    # 'nvv4l2camerasrc' 또는 'nvarguscamerasrc'를 사용하여 하드웨어 가속 활용
        return (
            "nvarguscamerasrc sensor-id=%d ! "
            "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                sensor_id,
                capture_width,
                capture_height,
                framerate,
                flip_method,
                display_width,
                display_height,
            )
        )
# ============================================================================
# 양자화 적용 함수
# ============================================================================
def apply_quantization(model, quant_choice, model_name):
    """모델에 양자화 적용"""
    quant_info = QUANTIZE_OPTIONS[quant_choice]
    quant_name = quant_info['name']
    
    if quant_choice == '1':
        # FP32 - 양자화 없음
        print("\n✅ FP32 (양자화 없음)")
        dtypes = torch.float32
        model = model.to(device)
        model.eval()
        return model, model_name
    
    elif quant_choice == '2':
        # FP16 - Half Precision (CPU에서는 제한적)
        print("\n📊 양자화 적용 중: {}".format(quant_name))
        try:
            dtypes = torch.float16
            model = model.half().to(device)
            model.eval()
            print("✅ FP16 변환 완료")
            model_name = "{} + FP16".format(model_name)
            return model, model_name
        except Exception as e:
            print("⚠️ FP16 변환 실패: {}".format(e))
            model = model.to(device)
            model.eval()
            return model, model_name
    
    
    model = model.to(device)
    model.eval()
    return model, model_name

# ============================================================================
# 캡션 생성 함수
# ============================================================================
# 캡션 생성 함수
# ============================================================================
def generate_caption_from_image(model, word_map, rev_word_map, frame):
    """이미지로부터 캡션 생성"""
    image_tensor = None
    try:
        # 모델을 디바이스로 이동
        model = model.to(device)
        model.eval()
        frame=frame.to(dtypes)
        # 이미지 전처리
        image_tensor = preprocess_image(frame)
        
        # 캡션 생성
        start_time = time.time()
        try:
            with torch.no_grad():
                # 메모리 안전성을 위해 배치 크기 = 1로 제한
                generated_words = model.generate(image_tensor, word_map, rev_word_map, max_len=50,device=device)
        except RuntimeError as e:
            print("경고: 추론 실패 - {}".format(e))
            gc.collect()
            return None, 0.0
        except Exception as e:
            print("경고: 예상 불가능한 오류 - {}".format(e))
            import traceback
            traceback.print_exc()
            gc.collect()
            return None, 0.0
        finally:
            # 이미지 텐서 메모리 해제
            if image_tensor is not None:
                del image_tensor
            gc.collect()
        
        inference_time = (time.time() - start_time) * 1000
        
        # 토큰 제거하고 문장으로 변환
        caption = ' '.join([w for w in generated_words if w not in ['<start>', '<end>', '<pad>', '<unk>']])
        
        return caption, inference_time
    except Exception as e:
        print("캡션 생성 오류: {}".format(e))
        import traceback
        traceback.print_exc()
        if image_tensor is not None:
            del image_tensor
        gc.collect()
        return None, 0.0

# ============================================================================
# 메인 실행 함수
# ============================================================================
def main():
    print("\n📊 Jetson Nano 이미지 캡셔닝 시스템")
    print("="*70)
    
    # 모델 선택
    model_choice = select_model()
    
    # 모델 로드
    model, word_map, rev_word_map, model_name = load_model(model_choice)
    if model is None:
        print("❌ 모델을 로드할 수 없습니다.")
        return
    
    # 양자화 선택 및 적용
    quant_choice = select_quantization()
    model, model_name = apply_quantization(model, quant_choice, model_name)
    
    try:
        # 더미 데이터 생성 (1, 3, 224, 224)
        dummy_input = torch.zeros(1, 3, 224, 224).to(device)
        if device.type == 'cuda':
            dummy_input = dummy_input.half() # FP16 모드라면

        # 강제로 한 번 실행시켜서 CUDA 커널을 깨움
        with torch.no_grad():
            # generate 함수가 아니라 encoder만 통과시켜도 효과 있음
            if hasattr(model, 'encoder'):
                _ = model.encoder(dummy_input)
        
        # GPU 동기화 (완료될 때까지 대기)
        if device.type == 'cuda':
            torch.cuda.synchronize()
            
        print("✅ 워밍업 완료! 이제 바로 캡션이 생성됩니다.")
    except Exception as e:
        print(f"⚠️ 워밍업 건너뜀: {e}")
    
    # 성능 모니터 생성
    try:
        monitor = PerformanceMonitor(model)
    except Exception as e:
        print("⚠️  성능 모니터 초기화 실패: {}".format(e))
        monitor = None

    # 카메라 초기화
    print("\n📹 카메라 초기화 중...")
    
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        cap=cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        return
    
    # 카메라 설정 최적화
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("\n" + "="*70)
    print("=== 이미지 캡셔닝 시스템 ({}) ===".format(model_name))
    print("="*70)
    print("\n⌨️  키보드 명령어:")
    print("  's' : 현재 프레임에서 캡션 생성 및 음성 출력")
    print("  'r' : 마지막 캡션 다시 듣기")
    print("  'p' : 성능 통계 출력")
    print("  'm' : 모델 변경")
    print("  'q' : 종료\n")
    
    last_caption = None
    is_processing = False
    current_model_name = model_name
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("카메라 읽기 실패")
            break
        
        # 메모리 기록
        if monitor:
            monitor.record_memory()
        
        # 메모리 모니터링 (5프레임마다)
        frame_count += 1
        if frame_count % 5 == 0:
            if monitor:
                current_mem = monitor.get_cpu_memory_mb()
                if current_mem > 2500:  # Jetson Nano 4GB 기준
                    print("⚠️  높은 메모리 사용: {:.0f}MB - 정리 중...".format(current_mem))
                    gc.collect()
        
        # 처리 중 표시
        if is_processing:
            cv2.putText(frame, "Processing...", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)
        
        # 모델 정보 표시
        cv2.rectangle(frame, (5, frame.shape[0] - 75), (550, frame.shape[0] - 5), (50, 50, 50), -1)
        cv2.putText(frame, "Model: {}".format(current_model_name[:40]), (10, frame.shape[0] - 52),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # 성능 지표 표시
        stats = monitor.get_stats()
        if stats:
            fps_text = "FPS: {:.1f}".format(stats['fps'])
            latency_text = "Latency: {:.1f}ms".format(stats['mean_latency_ms'])
            mem_text = "CPU: {:.0f}MB".format(stats['cpu_memory_mb'])
            
            cv2.putText(frame, fps_text, (10, frame.shape[0] - 32),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, latency_text, (10, frame.shape[0] - 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, mem_text, (frame.shape[1] - 250, frame.shape[0] - 32),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            if device.type in ['cuda', 'mps']:
                gpu_text = "GPU: {:.0f}MB".format(stats['gpu_memory_mb'])
                cv2.putText(frame, gpu_text, (frame.shape[1] - 250, frame.shape[0] - 12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 마지막 캡션 표시
        if last_caption and not is_processing:
            caption_y = 60
            max_width = frame.shape[1] - 20
            words = last_caption.split()
            line = ""
            line_num = 0
            
            for word in words:
                test_line = line + word + " "
                text_size = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                
                if text_size[0] > max_width:
                    text_size_actual = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(frame, (5, caption_y + line_num * 25 - 20), 
                                (15 + text_size_actual[0], caption_y + line_num * 25 + 5), 
                                (0, 0, 0), -1)
                    cv2.putText(frame, line, (10, caption_y + line_num * 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)
                    line = word + " "
                    line_num += 1
                else:
                    line = test_line
            
            if line:
                text_size_actual = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(frame, (5, caption_y + line_num * 25 - 20), 
                            (15 + text_size_actual[0], caption_y + line_num * 25 + 5), 
                            (0, 0, 0), -1)
                cv2.putText(frame, line, (10, caption_y + line_num * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)
        
        cv2.imshow('Image Captioning', frame)
        
        # 키 입력 처리
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n종료")
            break
            
        elif key == ord('s') and not is_processing:
            is_processing = True
            print("\n" + "="*70)
            print("캡션 생성 중...")
            
            caption, inf_time = generate_caption_from_image(model, word_map, rev_word_map, frame)
            monitor.record_inference(inf_time)
            
            if caption:
                last_caption = caption
                print("\n생성된 캡션: {}".format(caption))
                print("추론 시간: {:.2f}ms".format(inf_time))
                
                # 캡션 음성 출력
                speak_text_gtts(caption)
            else:
                print("캡션 생성 실패")
            
            print("="*70 + "\n")
            is_processing = False
            
        elif key == ord('r') and last_caption:
            print("\n🔊 마지막 캡션: \"{}\"".format(last_caption))
            speak_text_gtts(last_caption)
            
        elif key == ord('p'):
            monitor.print_stats()
            
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()