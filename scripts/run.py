import cv2
import torch
import numpy as np
import os
import threading
import tempfile
import time
from PIL import Image
from torchvision import transforms
from gtts import gTTS
import pygame
from src.muti_modal_model.model import MobileNetCaptioningModel

# ============================================================================
# 환경 설정
# ============================================================================
# 디바이스 선택
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"디바이스: {device}")

# 모델 경로 설정
MODEL_PATH = "models/lightweight_captioning_model.pth"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "lightweight_captioning_model.pth"

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

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
            print(f"TTS Error: {e}")
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
def load_model():
    """학습된 캡셔닝 모델 로드"""
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        return None, None, None
    
    try:
        print(f"📂 모델 로드 중: {MODEL_PATH}")
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            word_map = checkpoint.get('word_map')
            rev_word_map = checkpoint.get('rev_word_map')
            vocab_size = checkpoint.get('vocab_size')
            
            if word_map is None or rev_word_map is None:
                print("❌ 단어장 정보가 없습니다.")
                return None, None, None
            
            # 모델 생성
            model = MobileNetCaptioningModel(vocab_size=vocab_size, embed_dim=300).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            print(f"✅ 모델 로드 완료 (단어장 크기: {vocab_size})")
            return model, word_map, rev_word_map
        else:
            print("❌ 잘못된 모델 파일 형식입니다.")
            return None, None, None
            
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

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
        print(f"캡션 생성 오류: {e}")
        return None, 0.0

# ============================================================================
# 메인 실행 함수
# ============================================================================
def main():
    # 모델 로드
    model, word_map, rev_word_map = load_model()
    if model is None:
        print("모델을 로드할 수 없습니다. 학습을 먼저 실행하세요.")
        return
    
    # 카메라 초기화
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        return
    
    print("\n" + "="*70)
    print("=== 이미지 캡셔닝 실시간 실행 ===")
    print("="*70)
    print("\n키보드 명령어:")
    print("  's' : 현재 프레임에서 캡션 생성 및 음성 출력")
    print("  'r' : 마지막 캡션 다시 듣기")
    print("  'q' : 종료\n")
    
    last_caption = None
    is_processing = False
    inference_times = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("카메라 읽기 실패")
            break
        
        # 처리 중 표시
        if is_processing:
            cv2.putText(frame, "Processing...", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)
        
        # 모델 정보 표시
        cv2.rectangle(frame, (5, frame.shape[0] - 35), (500, frame.shape[0] - 5), (50, 50, 50), -1)
        cv2.putText(frame, "Image Captioning Model", (10, frame.shape[0] - 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # 평균 추론 시간 표시
        if inference_times:
            avg_inf_time = np.mean(inference_times[-30:])
            cv2.rectangle(frame, (frame.shape[1] - 200, frame.shape[0] - 35), 
                         (frame.shape[1] - 5, frame.shape[0] - 5), (50, 50, 50), -1)
            cv2.putText(frame, f"Inf: {avg_inf_time:.1f}ms", (frame.shape[1] - 190, frame.shape[0] - 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
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
            
            if caption:
                last_caption = caption
                inference_times.append(inf_time)
                print(f"\n생성된 캡션: {caption}")
                print(f"추론 시간: {inf_time:.2f}ms")
                
                # 캡션 음성 출력
                speak_text_gtts(caption)
            else:
                print("캡션 생성 실패")
            
            print("="*70 + "\n")
            is_processing = False
            
        elif key == ord('r') and last_caption:
            print(f"\n마지막 캡션: \"{last_caption}\"")
            speak_text_gtts(last_caption)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()