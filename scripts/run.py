import cv2
import torch
import numpy as np
import os
import threading
import tempfile
import time
import psutil
from PIL import Image
from torchvision import transforms
from gtts import gTTS
import pygame
from src.muti_modal_model.model import MobileNetCaptioningModel
from src.utils.quantization_utils import apply_dynamic_quantization

# ============================================================================
# 환경 설정
# ============================================================================
# 디바이스 선택
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("디바이스: {}".format(device))

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

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# ============================================================================
# 성능 모니터링 클래스
# ============================================================================
class PerformanceMonitor:
    """모델 성능 모니터링"""
    def __init__(self):
        self.inference_times = []
        self.memory_usage = []
        self.gpu_memory = []
        self.process = psutil.Process(os.getpid())
    
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
        print(f"⏱️  추론 시간 (Latency):")
        print(f"    • 평균: {stats['mean_latency_ms']:.2f} ms")
        print(f"    • 중앙값: {stats['median_latency_ms']:.2f} ms")
        print(f"    • 최소/최대: {stats['min_latency_ms']:.2f} / {stats['max_latency_ms']:.2f} ms")
        print(f"    • 표준편차: {stats['std_latency_ms']:.2f} ms")
        print(f"\n🎬 처리 속도 (Throughput):")
        print(f"    • FPS: {stats['fps']:.1f} frame/sec")
        print(f"    • 1프레임 처리: {stats['mean_latency_ms']:.2f} ms")
        print(f"\n💾 메모리 사용량:")
        print(f"    • CPU: {stats['cpu_memory_mb']:.1f} MB")
        if device.type in ['cuda', 'mps']:
            print(f"    • GPU: {stats['gpu_memory_mb']:.1f} MB")
        print(f"\n📊 누적 통계:")
        print(f"    • 총 추론 횟수: {stats['total_inferences']}회")
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
                print(f"❌ 모델 파일을 찾을 수 없습니다: {model_info['path']}")
                return None, None, None, None
        else:
            print("❌ 모델 파일을 찾을 수 없습니다: {}".format(model_path))
            return None, None, None, None
    
    try:
        print("\n📂 모델 로드 중: {}".format(model_path))
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
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
            
            print(f"   📐 감지된 모델 구조:")
            print("      • Decoder Dim: {}".format(decoder_dim))
            print("      • Attention Dim: {}".format(attention_dim))
            
            # 올바른 크기로 모델 생성
            model = MobileNetCaptioningModel(
                vocab_size=vocab_size, 
                embed_dim=300,
                decoder_dim=decoder_dim,
                attention_dim=attention_dim
            ).to(device)
            
            # state_dict 로드 (strict=False로 호환되는 레이어만 로드)
            try:
                model.load_state_dict(state_dict, strict=False)
                print(f"✅ 모델 상태 로드 완료")
            except Exception as e:
                print("⚠️  상태 로드 중 경고: {}".format(e))
            
            model.eval()
            
            model_name = model_info['name']
            
            # 모델 크기 계산
            param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers()) / 1024 / 1024
            total_params = sum(p.numel() for p in model.parameters())
            
            print(f"\n✅ 모델 로드 완료")
            print("   모델: {}".format(model_name))
            print("   단어장 크기: {}".format(vocab_size))
            print("   총 파라미터: {}".format(total_params:,))
            print("   모델 크기: {} MB".format(param_size + buffer_size:.2f))
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
# 캡션 생성 함수
# ============================================================================
def generate_caption_from_image(model, word_map, rev_word_map, frame):
    """이미지로부터 캡션 생성"""
    try:
        # OpenCV BGR을 RGB로 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # 전처리
        image_tensor = transform(pil_image).unsqueeze(0).to(device)
        
        # 캡션 생성
        start_time = time.time()
        with torch.no_grad():
            generated_words = model.generate(image_tensor, word_map, rev_word_map, max_len=50)
        
        inference_time = (time.time() - start_time) * 1000
        
        # 토큰 제거하고 문장으로 변환
        caption = ' '.join([w for w in generated_words if w not in ['<start>', '<end>', '<pad>', '<unk>']])
        
        return caption, inference_time
    except Exception as e:
        print("캡션 생성 오류: {}".format(e))
        return None, 0.0

# ============================================================================
# 메인 실행 함수
# ============================================================================
def main():
    # 성능 모니터 생성
    monitor = PerformanceMonitor()
    
    # 모델 선택
    model_choice = select_model()
    
    # 모델 로드
    model, word_map, rev_word_map, model_name = load_model(model_choice)
    if model is None:
        print("❌ 모델을 로드할 수 없습니다.")
        return
    
    # 카메라 초기화
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        return
    
    print("\n" + "="*70)
    print("=== 이미지 캡셔닝 실시간 실행 ({}) ===".format(model_name))
    print("="*70)
    print("\n⌨️  키보드 명령어:")
    print("  's' : 현재 프레임에서 캡션 생성 및 음성 출력")
    print("  'r' : 마지막 캡션 다시 듣기")
    print("  'p' : 성능 통계 출력 (JTOPS 스타일)")
    print("  'm' : 모델 변경")
    print("  'q' : 종료\n")
    
    last_caption = None
    is_processing = False
    current_model_name = model_name
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("카메라 읽기 실패")
            break
        
        # 메모리 기록
        monitor.record_memory()
        
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
            fps_text = f"FPS: {stats['fps']:.1f}"
            latency_text = f"Latency: {stats['mean_latency_ms']:.1f}ms"
            mem_text = f"CPU: {stats['cpu_memory_mb']:.0f}MB"
            
            cv2.putText(frame, fps_text, (10, frame.shape[0] - 32),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, latency_text, (10, frame.shape[0] - 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, mem_text, (frame.shape[1] - 250, frame.shape[0] - 32),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            if device.type in ['cuda', 'mps']:
                gpu_text = f"GPU: {stats['gpu_memory_mb']:.0f}MB"
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
                print("추론 시간: {}ms".format(inf_time:.2f))
                
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
            
            current_model_name = model_name
            last_caption = None
            monitor = PerformanceMonitor()  # 새 모니터 생성
            
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("❌ 카메라를 열 수 없습니다.")
                return
            
            print("\n✅ {} 모델로 변경되었습니다.\n".format(model_name))
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()