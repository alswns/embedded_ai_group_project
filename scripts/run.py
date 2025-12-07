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

try:
    from src.muti_modal_model.model import MobileNetCaptioningModel
    from src.utils.quantization_utils import apply_dynamic_quantization
    print("   ✅ 프로젝트 모듈 로드", file=sys.stderr)
except ImportError as e:
    print("❌ 프로젝트 모듈 오류: {}".format(e), file=sys.stderr)
    sys.exit(1)

print("✅ 모든 모듈 로드 완료", file=sys.stderr)

# ============================================================================
# 환경 설정 (CRITICAL - 크래시 방지)
# ============================================================================
print("⚙️  환경 설정 중...", file=sys.stderr)

# GPU 완전 비활성화 (CPU 전용)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

# CPU 스레드 제한
torch.set_num_threads(2)
torch.set_num_interop_threads(1)

# 디바이스 설정 (강제 CPU)
device = torch.device("cpu")
print("📍 디바이스: CPU (GPU 비활성화됨)", file=sys.stderr)

# ============================================================================
# 이미지 전처리 함수 (torchvision 대체)
# ============================================================================
def preprocess_image_manual(frame):
    """torchvision 없이 이미지 전처리"""
    # BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    
    # 리사이즈
    pil_image = pil_image.resize((224, 224), Image.BILINEAR)
    
    # numpy array
    image_array = np.array(pil_image, dtype=np.float32) / 255.0
    
    # 정규화
    image_array -= np.array([0.485, 0.456, 0.406], dtype=np.float32)
    image_array /= np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    # CHW 형식
    image_array = np.transpose(image_array, (2, 0, 1))
    
    # 텐서로 변환
    image_tensor = torch.from_numpy(image_array).float().unsqueeze(0)
    
    return image_tensor

preprocess_image = preprocess_image_manual

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
    '3': {'name': 'INT8 (Dynamic Quantization)', 'enabled': True}
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
        
        # CPU에서 로드 (메모리 안전) - Python/PyTorch 버전 호환성
        try:
            # Python 3.11+: weights_only 파라미터 필요
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        except TypeError:
            # Python 3.6-3.10: weights_only 파라미터 미지원
            checkpoint = torch.load(model_path, map_location='cpu')
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            word_map = checkpoint.get('word_map')
            rev_word_map = checkpoint.get('rev_word_map')
            vocab_size = checkpoint.get('vocab_size')
            
            if word_map is None or rev_word_map is None:
                print("❌ 단어장 정보가 없습니다.")
                return None, None, None, None
            
            # 체크포인트에서 모델 크기 정보 추출
            state_dict = checkpoint['model_state_dict']
            
            decoder_dim = checkpoint.get('decoder_dim', 512)
            attention_dim = checkpoint.get('attention_dim', 256)
            
            # state_dict에서 크기 정보가 없으면 자동 추출
            if 'decoder.decode_step.weight_ih' in state_dict:
                decoder_dim = state_dict['decoder.decode_step.weight_ih'].shape[0] // 3
            
            if 'decoder.encoder_att.weight' in state_dict:
                attention_dim = state_dict['decoder.encoder_att.weight'].shape[0]
            
            print("   📐 감지된 모델 구조:")
            print("      • Decoder Dim: {}".format(decoder_dim))
            print("      • Attention Dim: {}".format(attention_dim))
            
            # 올바른 크기로 모델 생성 (CPU에서만)
            try:
                model = MobileNetCaptioningModel(
                    vocab_size=vocab_size, 
                    embed_dim=300,
                    decoder_dim=decoder_dim,
                    attention_dim=attention_dim
                )
                model = model.to(device)
            except Exception as e:
                print("❌ 모델 생성 실패: {}".format(e))
                return None, None, None, None
            
            # state_dict 로드 (strict=False로 호환되는 레이어만 로드)
            try:
                model.load_state_dict(state_dict, strict=False)
                print("✅ 모델 상태 로드 완료")
            except Exception as e:
                print("⚠️  상태 로드 중 경고: {}".format(e))
                import traceback
                traceback.print_exc()
            
            # 메모리 정리
            del checkpoint, state_dict
            gc.collect()
            
            model.eval()
            
            # 모델 to CPU 명시
            try:
                model = model.cpu()
                model.eval()
            except:
                pass
            
            model_name = model_info['name']
            
            print("\n✅ 모델 로드 완료")
            print("   모델: {}".format(model_name))
            print("   경로: {}".format(model_path))
            
            return model, word_map, rev_word_map, model_name
        else:
            print("❌ 잘못된 모델 파일 형식입니다.")
            return None, None, None, None
            
    except Exception as e:
        print("❌ 모델 로드 실패: {}".format(e))
        import traceback
        traceback.print_exc()
        return None, None, None, None

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
        model = model.cpu()
        model.eval()
        return model, model_name
    
    elif quant_choice == '2':
        # FP16 - Half Precision (CPU에서는 제한적)
        print("\n📊 양자화 적용 중: {}".format(quant_name))
        try:
            # CPU에서는 FP16이 지원되지 않으므로 FP32 유지
            print("⚠️  CPU에서는 FP16이 지원되지 않습니다. FP32로 유지합니다.")
            model = model.cpu()
            model.eval()
            return model, model_name
        except Exception as e:
            print("⚠️ FP16 변환 실패: {}".format(e))
            model = model.cpu()
            model.eval()
            return model, model_name
    
    elif quant_choice == '3':
        # INT8 - Dynamic Quantization
        print("\n📊 양자화 적용 중: {}".format(quant_name))
        try:
            # CPU 기반 INT8 양자화 (안전 버전)
            model = model.cpu()
            model.eval()
            
            # Dynamic Quantization 적용 (CPU 안전)
            try:
                quantized_model = apply_dynamic_quantization(model)
                print("✅ INT8 양자화 완료")
                model_name = "{} + INT8".format(model_name)
                return quantized_model, model_name
            except Exception as e2:
                print("⚠️  INT8 적용 실패, FP32로 진행합니다: {}".format(e2))
                return model, model_name
        except Exception as e:
            print("⚠️ INT8 양자화 실패: {}. 원본 모델로 계속합니다.".format(e))
            model = model.cpu()
            model.eval()
            return model, model_name
    
    model = model.cpu()
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
        # 모델을 CPU로 이동
        model = model.cpu()
        model.eval()
        
        # 이미지 전처리
        image_tensor = preprocess_image(frame)
        
        # 캡션 생성
        start_time = time.time()
        try:
            with torch.no_grad():
                # 메모리 안전성을 위해 배치 크기 = 1로 제한
                generated_words = model.generate(image_tensor, word_map, rev_word_map, max_len=50)
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
    
    # CPU 모드 명시적 설정
    model = model.cpu()
    model.eval()
    
    # 성능 모니터 생성
    try:
        monitor = PerformanceMonitor(model)
    except Exception as e:
        print("⚠️  성능 모니터 초기화 실패: {}".format(e))
        monitor = None

    # 카메라 초기화
    print("\n📹 카메라 초기화 중...")
    cap = cv2.VideoCapture(0)
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
            
        elif key == ord('m'):
            print("\n모델을 변경합니다...")
            cap.release()
            cv2.destroyAllWindows()
            
            model_choice = select_model()
            model, word_map, rev_word_map, model_name = load_model(model_choice)
            
            if model is None:
                print("❌ 모델을 로드할 수 없습니다.")
                return
            
            # 양자화 선택 및 적용
            quant_choice = select_quantization()
            model, model_name = apply_quantization(model, quant_choice, model_name)
            
            current_model_name = model_name
            last_caption = None
            monitor = PerformanceMonitor(model)  # 새 모니터 생성
            
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("❌ 카메라를 열 수 없습니다.")
                return
            
            print("\n✅ {} 모델로 변경되었습니다.\n".format(model_name))
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()