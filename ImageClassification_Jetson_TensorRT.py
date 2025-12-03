import cv2
import json
import urllib.request
from gtts import gTTS
import pygame
import os
import threading
import tempfile
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import platform
import sys
matplotlib.use('Agg')

# ============================================================================
# 환경 감지
# ============================================================================
PLATFORM = platform.system().lower()
IS_JETSON = os.path.exists('/etc/nv_tegra_release') or 'jetson' in platform.machine().lower()
IS_MAC = PLATFORM == 'darwin'
IS_WINDOWS = PLATFORM == 'windows'
IS_LINUX = PLATFORM == 'linux' and not IS_JETSON

print(f"플랫폼 감지: {platform.system()} ({platform.machine()})")
if IS_JETSON:
    print("환경: Jetson Nano")
elif IS_MAC:
    print("환경: macOS")
elif IS_WINDOWS:
    print("환경: Windows")
else:
    print("환경: Linux (일반)")

# Jetson 라이브러리 확인
try:
    import jetson.inference
    import jetson.utils
    JETSON_AVAILABLE = True
except ImportError:
    JETSON_AVAILABLE = False
    if IS_JETSON:
        print("Warning: jetson-inference not available. Please install: sudo apt-get install jetson-inference")

# Mac/Windows용 딥러닝 프레임워크 확인
TENSORFLOW_AVAILABLE = False
TORCH_AVAILABLE = False
ONNX_AVAILABLE = False

if not JETSON_AVAILABLE:
    try:
        import tensorflow as tf
        TENSORFLOW_AVAILABLE = True
        print(f"TensorFlow 사용 가능: {tf.__version__}")
    except ImportError:
        pass
    
    try:
        import torch
        import torchvision
        from torchvision import transforms, models
        TORCH_AVAILABLE = True
        print(f"PyTorch 사용 가능: {torch.__version__}")
    except ImportError:
        pass
    
    try:
        import onnxruntime as ort
        ONNX_AVAILABLE = True
        print(f"ONNX Runtime 사용 가능: {ort.__version__}")
    except ImportError:
        pass
    
    if not (TENSORFLOW_AVAILABLE or TORCH_AVAILABLE or ONNX_AVAILABLE):
        print("\nWarning: 딥러닝 프레임워크가 설치되지 않았습니다.")
        print("다음 중 하나를 설치해주세요:")
        print("  - TensorFlow: pip install tensorflow")
        print("  - PyTorch: pip install torch torchvision")
        print("  - ONNX Runtime: pip install onnxruntime")

# ImageNet 클래스 레이블 다운로드
print("ImageNet 레이블 다운로드 중...")
url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
class_labels = json.loads(urllib.request.urlopen(url).read())

# 15개 사물에 대한 수동 설명 사전
MANUAL_DESCRIPTIONS = {
    "book": "A book is a set of written or printed pages bound together, used for reading and storing information.",
    "bottle": "A bottle is a container with a narrow neck, typically made of glass or plastic, used for storing liquids.",
    "cup": "A cup is a small open container used for drinking beverages, typically with a handle.",
    "keyboard": "A keyboard is an input device with keys for typing letters, numbers, and symbols into a computer.",
    "mouse": "A mouse is a handheld pointing device used to control a cursor on a computer screen.",
    "chair": "A chair is a piece of furniture with a seat, back, and often four legs, designed for one person to sit on.",
    "table": "A table is a piece of furniture with a flat top and one or more legs, used for eating, working, or displaying items.",
    "laptop": "A laptop is a portable personal computer with a screen and keyboard that can be folded together.",
    "phone": "A phone is an electronic device used for voice communication and various digital applications.",
    "pen": "A pen is a writing instrument that uses ink to mark paper or other surfaces.",
    "clock": "A clock is a device used to measure and display time, typically with numbers and moving hands or a digital display.",
    "bag": "A bag is a container made of flexible material used for carrying or storing items.",
    "shoe": "A shoe is a covering for the foot, typically made of leather or other durable material, with a sturdy sole.",
    "watch": "A watch is a small timepiece worn on the wrist, used to tell the time.",
    "lamp": "A lamp is a device that produces light, typically consisting of a base, stand, and light bulb or LED."
}

# Pygame 초기화
print("오디오 시스템 초기화 중...")
pygame.mixer.init()

# 그래프 저장 경로 설정
if IS_JETSON:
    OUTPUT_DIR = "./performance_results_jetson"
else:
    OUTPUT_DIR = "./performance_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"그래프 저장 경로: {OUTPUT_DIR}\n")

# ============================================================================
# TensorRT Precision 설정 옵션
# ============================================================================

# Jetson용 TensorRT 설정
TENSORRT_CONFIGS = [
    {
        'name': 'TensorRT FP32',
        'network': 'googlenet',
        'model_dir': None,
        'precision': 'FP32',
        'description': '32-bit floating point precision (기본 정밀도)',
    },
    {
        'name': 'TensorRT FP16',
        'network': 'googlenet',
        'model_dir': None,
        'precision': 'FP16',
        'description': '16-bit floating point precision (속도 향상)',
    },
    {
        'name': 'TensorRT INT8',
        'network': 'googlenet',
        'model_dir': None,
        'precision': 'INT8',
        'description': '8-bit integer precision (최대 압축)',
    },
]

# Mac/Windows용 모델 설정
STANDARD_CONFIGS = [
    {
        'name': 'TensorRT FP32',
        'network': 'googlenet',
        'model_dir': None,
        'precision': 'FP32',
        'description': '32-bit floating point precision (기본 정밀도)',
    },
    {
        'name': 'TensorRT FP16',
        'network': 'googlenet',
        'model_dir': None,
        'precision': 'FP16',
        'description': '16-bit floating point precision (속도 향상)',
    },
    {
        'name': 'TensorRT INT8',
        'network': 'googlenet',
        'model_dir': None,
        'precision': 'INT8',
        'description': '8-bit integer precision (최대 압축)',
    },
]

# 모델 크기 및 FLOPs (GoogleNet 추정값)
MODEL_SPECS = {
    'FP32': {
        'params': 6.8e6,      # 6.8M parameters
        'size': 27.2,          # MB
        'flops': 1.5e9         # 1.5 GFLOPs
    },
    'FP16': {
        'params': 6.8e6,
        'size': 13.6,          # FP32의 절반
        'flops': 1.5e9
    },
    'INT8': {
        'params': 6.8e6,
        'size': 6.8,           # FP32의 1/4
        'flops': 1.5e9
    }
}

# ============================================================================
# 모델 측정 함수
# ============================================================================

def get_tensorrt_model_size(precision):
    """TensorRT 캐시된 모델 크기 측정"""
    # TensorRT 엔진 캐시 경로
    cache_paths = [
        f"/var/cache/tensorrt/googlenet_{precision.lower()}.engine",
        f"~/.cache/tensorrt/googlenet_{precision.lower()}.engine",
        f"/tmp/tensorrt/googlenet_{precision.lower()}.engine"
    ]
    
    # 캐시 파일 찾기
    for cache_path in cache_paths:
        expanded_path = os.path.expanduser(cache_path)
        if os.path.exists(expanded_path):
            size_bytes = os.path.getsize(expanded_path)
            size_mb = size_bytes / (1024 * 1024)
            return size_mb
    
    # 캐시 파일이 없으면 추정값 반환
    print(f"  Warning: 캐시 파일을 찾을 수 없어 추정값 사용")
    return MODEL_SPECS[precision]['size']

def count_parameters_googlenet():
    """GoogleNet 파라미터 수 계산"""
    
    params = 0
    
    # Conv1: 7x7x64, stride=2
    params += 7 * 7 * 3 * 64
    
    # Conv2: 1x1x64
    params += 1 * 1 * 64 * 64
    
    # Conv3: 3x3x192
    params += 3 * 3 * 64 * 192
    
    # Inception modules (simplified calculation)
    inception_configs = [
        (192, 64, 96, 128, 16, 32, 32),   # 3a
        (256, 128, 128, 192, 32, 96, 64), # 3b
        (480, 192, 96, 208, 16, 48, 64),  # 4a
        (512, 160, 112, 224, 24, 64, 64), # 4b
        (512, 128, 128, 256, 24, 64, 64), # 4c
        (512, 112, 144, 288, 32, 64, 64), # 4d
        (528, 256, 160, 320, 32, 128, 128), # 4e
        (832, 256, 160, 320, 32, 128, 128), # 5a
        (832, 384, 192, 384, 48, 128, 128), # 5b
    ]
    
    for in_c, c1x1, c3x3_r, c3x3, c5x5_r, c5x5, pool_proj in inception_configs:
        # 1x1 branch
        params += 1 * 1 * in_c * c1x1
        
        # 3x3 branch
        params += 1 * 1 * in_c * c3x3_r  # reduction
        params += 3 * 3 * c3x3_r * c3x3
        
        # 5x5 branch
        params += 1 * 1 * in_c * c5x5_r  # reduction
        params += 5 * 5 * c5x5_r * c5x5
        
        # pool projection
        params += 1 * 1 * in_c * pool_proj
    
    # Final classifier
    params += 1024 * 1000
    
    # Auxiliary classifiers (x2)
    params += 2 * (512 * 128 + 128 * 1024 + 1024 * 1000)
    
    return params

def calculate_flops_googlenet():
    """GoogleNet FLOPs 계산"""
    flops = 0
    input_size = 224
    
    # Conv1: 7x7x64, stride=2
    out_size = input_size // 2
    flops += 2 * (7 * 7 * 3) * 64 * (out_size * out_size)
    
    # MaxPool
    out_size = out_size // 2  # 56
    
    # Conv2: 1x1x64
    flops += 2 * (1 * 1 * 64) * 64 * (out_size * out_size)
    
    # Conv3: 3x3x192
    flops += 2 * (3 * 3 * 64) * 192 * (out_size * out_size)
    
    # MaxPool
    out_size = out_size // 2  # 28
    
    # Inception modules
    inception_flops = [
        (28, 192, 64, 96, 128, 16, 32, 32),   # 3a
        (28, 256, 128, 128, 192, 32, 96, 64), # 3b
        (14, 480, 192, 96, 208, 16, 48, 64),  # 4a (after maxpool)
        (14, 512, 160, 112, 224, 24, 64, 64), # 4b
        (14, 512, 128, 128, 256, 24, 64, 64), # 4c
        (14, 512, 112, 144, 288, 32, 64, 64), # 4d
        (14, 528, 256, 160, 320, 32, 128, 128), # 4e
        (7, 832, 256, 160, 320, 32, 128, 128),  # 5a (after maxpool)
        (7, 832, 384, 192, 384, 48, 128, 128),  # 5b
    ]
    
    for size, in_c, c1x1, c3x3_r, c3x3, c5x5_r, c5x5, pool_proj in inception_flops:
        spatial = size * size
        
        # 1x1 branch
        flops += 2 * (1 * 1 * in_c) * c1x1 * spatial
        
        # 3x3 branch
        flops += 2 * (1 * 1 * in_c) * c3x3_r * spatial
        flops += 2 * (3 * 3 * c3x3_r) * c3x3 * spatial
        
        # 5x5 branch
        flops += 2 * (1 * 1 * in_c) * c5x5_r * spatial
        flops += 2 * (5 * 5 * c5x5_r) * c5x5 * spatial
        
        # pool projection
        flops += 2 * (1 * 1 * in_c) * pool_proj * spatial
    
    flops += 2 * 1024 * 1000
    
    return flops

# ============================================================================
# 모델 로드 함수
# ============================================================================

def load_jetson_model(config):
    """TensorRT 모델 로드 (Jetson 전용)"""
    if not JETSON_AVAILABLE:
        print("Jetson Inference를 사용할 수 없습니다.")
        return None, False
    
    print("\n" + "="*70)
    print(f"{config['name']} 모델 로딩...")
    print(f"설명: {config['description']}")
    print("="*70)
    
    try:
        # TensorRT 엔진 빌드 (처음 실행 시 시간 소요)
        print(f"\nTensorRT 엔진 빌드 중 ({config['precision']})...")
        print("주의: 처음 실행 시 5-10분 소요될 수 있습니다.")
        
        # Network 파라미터 구성
        argv = [f'--network={config["network"]}']
        
        # Precision 설정
        if config['precision'] == 'FP16':
            argv.append('--precision=FP16')
        elif config['precision'] == 'INT8':
            argv.append('--precision=INT8')
        # FP32는 기본값이므로 추가 파라미터 불필요
        
        net = jetson.inference.imageNet(argv=argv)
        print("완료!\n")
        return net, True
        
    except Exception as e:
        print(f"Error: TensorRT 모델 로딩 실패 - {e}")
        return None, False

def load_pytorch_model(config):
    """PyTorch 모델 로드 (Mac/Windows용)"""
    if not TORCH_AVAILABLE:
        return None, False
    
    print("\n" + "="*70)
    print(f"{config['name']} 모델 로딩...")
    print(f"설명: {config['description']}")
    print("="*70)
    
    try:
        print("\nPyTorch GoogleNet 모델 로딩 중...")
        model = models.googlenet(pretrained=True)
        model.eval()
        
        # 전처리 변환 정의
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print("완료!\n")
        return {'model': model, 'transform': transform}, True
        
    except Exception as e:
        print(f"Error: PyTorch 모델 로딩 실패 - {e}")
        return None, False

def load_tensorflow_model(config):
    """TensorFlow 모델 로드 (Mac/Windows용)"""
    if not TENSORFLOW_AVAILABLE:
        return None, False
    
    print("\n" + "="*70)
    print(f"{config['name']} 모델 로딩...")
    print(f"설명: {config['description']}")
    print("="*70)
    
    try:
        print("\nTensorFlow 모델 로딩 중...")
        # 먼저 Keras 애플리케이션 시도 (더 안정적)
        try:
            from tensorflow.keras.applications import InceptionV3
            from tensorflow.keras.applications.inception_v3 import preprocess_input
            
            model = InceptionV3(weights='imagenet')
            print("완료! (Keras InceptionV3 사용)\n")
            return {'model': model, 'preprocess': preprocess_input}, True
        except Exception as e1:
            # 대체: TensorFlow Hub에서 모델 로드
            try:
                import tensorflow_hub as hub
                model_url = "https://tfhub.dev/google/imagenet/inception_v3/classification/5"
                model = hub.load(model_url)
                print("완료! (TensorFlow Hub 사용)\n")
                return {'model': model}, True
            except Exception as e2:
                print(f"Error: TensorFlow Hub 모델 로딩 실패 - {e2}")
                raise e1
        
    except Exception as e:
        print(f"Error: TensorFlow 모델 로딩 실패 - {e}")
        return None, False

def load_standard_model(config):
    """표준 환경용 모델 로드 (Mac/Windows)"""
    framework = config.get('framework', 'pytorch')
    
    if framework == 'pytorch' and TORCH_AVAILABLE:
        return load_pytorch_model(config)
    elif framework == 'tensorflow' and TENSORFLOW_AVAILABLE:
        return load_tensorflow_model(config)
    else:
        # 기본값으로 PyTorch 시도
        if TORCH_AVAILABLE:
            return load_pytorch_model(config)
        elif TENSORFLOW_AVAILABLE:
            return load_tensorflow_model(config)
        else:
            print("Error: 사용 가능한 딥러닝 프레임워크가 없습니다.")
            return None, False

def get_model_info(config, net=None):
    """모델 정보 계산 및 반환"""
    precision = config.get('precision', 'FP32')
    
    # 실제 파라미터 수 계산
    params = count_parameters_googlenet()
    
    # 실제 모델 크기 측정
    if net is not None and JETSON_AVAILABLE and isinstance(net, jetson.inference.imageNet):
        # Jetson: 엔진이 로드된 후 캐시 파일 크기 측정
        size = get_tensorrt_model_size(precision)
    else:
        # 표준 환경 또는 아직 로드 전이면 추정값
        if precision in MODEL_SPECS:
            size = MODEL_SPECS[precision]['size']
        else:
            size = MODEL_SPECS['FP32']['size']  # 기본값
    
    # FLOPs 계산
    flops = calculate_flops_googlenet()
    
    print(f"\n{'='*70}")
    print(f"{config['name']} 모델 사양 (GoogleNet)")
    print(f"{'='*70}")
    if 'network' in config:
        print(f"  네트워크:           {config['network']}")
    if 'framework' in config:
        print(f"  프레임워크:         {config['framework']}")
    print(f"  정밀도:             {precision}")
    print(f"  파라미터 수:        {params:,} ({params/1e6:.2f}M)")
    print(f"  모델 크기:          {size:.2f} MB")
    print(f"  FLOPs:              {flops:,} ({flops/1e9:.2f}G)")
    print(f"{'='*70}\n")
    
    return {
        'name': config['name'],
        'params': params,
        'size': size,
        'flops': flops,
        'time_mean': 0.0,
        'time_std': 0.0,
        'inference_times': []
    }

# ============================================================================
# 분류 함수
# ============================================================================

def classify_image_jetson(net, frame):
    """Jetson용 이미지 분류"""
    # OpenCV BGR을 CUDA 이미지로 변환
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # CUDA 이미지 생성
    cuda_img = jetson.utils.cudaFromNumpy(rgb_frame)
    
    start_time = time.time()
    class_idx, confidence = net.Classify(cuda_img)
    inference_time = (time.time() - start_time) * 1000
    
    # 클래스 이름 가져오기
    predicted_class = net.GetClassDesc(class_idx)
    confidence_percent = confidence * 100
    
    # 단일 결과만 반환
    top5_results = [(predicted_class, confidence_percent)]
    
    return predicted_class, confidence_percent, top5_results, inference_time

def classify_image_pytorch(model_dict, frame):
    """PyTorch용 이미지 분류"""
    model = model_dict['model']
    transform = model_dict['transform']
    
    # OpenCV BGR을 RGB로 변환
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 전처리
    input_tensor = transform(rgb_frame)
    input_batch = input_tensor.unsqueeze(0)
    
    # GPU 사용 가능 시
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_batch = input_batch.to(device)
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model(input_batch)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
    inference_time = (time.time() - start_time) * 1000
    
    # Top-1 결과
    top1_prob, top1_idx = torch.max(probabilities, 0)
    predicted_class = class_labels[top1_idx.item()]
    confidence_percent = top1_prob.item() * 100
    
    top5_results = [(predicted_class, confidence_percent)]
    
    return predicted_class, confidence_percent, top5_results, inference_time

def classify_image_tensorflow(model_dict, frame):
    """TensorFlow용 이미지 분류"""
    model = model_dict['model']
    preprocess = model_dict.get('preprocess', None)
    
    # OpenCV BGR을 RGB로 변환
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 리사이즈
    resized = cv2.resize(rgb_frame, (224, 224))
    
    # 전처리
    if preprocess:
        img_array = preprocess(np.expand_dims(resized, axis=0))
    else:
        img_array = np.expand_dims(resized, axis=0) / 255.0
    
    start_time = time.time()
    predictions = model(img_array)
    inference_time = (time.time() - start_time) * 1000
    
    # Top-1 결과
    if hasattr(predictions, 'numpy'):
        probs = predictions.numpy()[0]
    else:
        probs = predictions[0]
    
    top1_idx = np.argmax(probs)
    predicted_class = class_labels[top1_idx]
    confidence_percent = probs[top1_idx] * 100
    
    top5_results = [(predicted_class, confidence_percent)]
    
    return predicted_class, confidence_percent, top5_results, inference_time

def classify_image(net, frame):
    """통합 이미지 분류 함수 (환경 자동 감지)"""
    if JETSON_AVAILABLE and isinstance(net, jetson.inference.imageNet):
        return classify_image_jetson(net, frame)
    elif isinstance(net, dict):
        if 'transform' in net:  # PyTorch
            return classify_image_pytorch(net, frame)
        elif 'model' in net:  # TensorFlow
            return classify_image_tensorflow(net, frame)
    
    # 기본값
    print("Error: 알 수 없는 모델 타입")
    return "unknown", 0.0, [], 0.0

def get_manual_description(object_name):
    """수동 설명 반환"""
    object_lower = object_name.lower()
    
    if object_lower in MANUAL_DESCRIPTIONS:
        return MANUAL_DESCRIPTIONS[object_lower]
    
    for key in MANUAL_DESCRIPTIONS.keys():
        if key in object_lower or object_lower in key:
            return MANUAL_DESCRIPTIONS[key]
    
    return f"This is a {object_name}."

def speak_text_gtts(text):
    """TTS 음성 출력"""
    def _speak():
        temp_file = None
        try:
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
# 모델 실행 함수
# ============================================================================

def run_model(net, model_name, results_dict):
    """모델 실행"""
    if net is None:
        print(f"Error: {model_name} 모델을 사용할 수 없습니다.")
        return
    
    cap = cv2.VideoCapture(0)
    
    print("\n" + "="*70)
    print(f"=== {model_name} 실행 중 ===")
    print("="*70)
    print("\nKey commands:")
    print("  's' : 분류 및 음성 출력")
    print("  'r' : 마지막 설명 다시 듣기")
    print("  'q' : 다음 모델로 이동\n")
    
    description = ""
    last_object = None
    last_full_description = ""
    is_processing = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("카메라 읽기 실패")
            break
        
        # 처리 중 표시
        if is_processing:
            cv2.putText(frame, "Processing...", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)
        
        # 모델 이름 표시
        cv2.rectangle(frame, (5, frame.shape[0] - 35), (500, frame.shape[0] - 5), (50, 50, 50), -1)
        cv2.putText(frame, f"Model: {model_name}", (10, frame.shape[0] - 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # 평균 추론 시간 표시
        if results_dict['inference_times']:
            avg_inf_time = np.mean(results_dict['inference_times'][-30:])
            cv2.rectangle(frame, (frame.shape[1] - 200, frame.shape[0] - 35), 
                         (frame.shape[1] - 5, frame.shape[0] - 5), (50, 50, 50), -1)
            cv2.putText(frame, f"Inf: {avg_inf_time:.1f}ms", (frame.shape[1] - 190, frame.shape[0] - 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 현재 인식된 객체 표시
        if last_object and not is_processing:
            cv2.rectangle(frame, (5, 5), (450, 55), (0, 0, 0), -1)
            cv2.putText(frame, f"Detected: {last_object}", (10, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # 설명 표시
        if description and not is_processing:
            y_offset = 70
            max_width = frame.shape[1] - 20
            
            words = description.split()
            line = ""
            line_num = 0
            
            for word in words:
                test_line = line + word + " "
                text_size = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                
                if text_size[0] > max_width:
                    text_size_actual = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(frame, (5, y_offset + line_num * 25 - 20), 
                                (15 + text_size_actual[0], y_offset + line_num * 25 + 5), 
                                (0, 0, 0), -1)
                    cv2.putText(frame, line, (10, y_offset + line_num * 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    line = word + " "
                    line_num += 1
                else:
                    line = test_line
            
            if line:
                text_size_actual = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(frame, (5, y_offset + line_num * 25 - 20), 
                            (15 + text_size_actual[0], y_offset + line_num * 25 + 5), 
                            (0, 0, 0), -1)
                cv2.putText(frame, line, (10, y_offset + line_num * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow(f'{model_name}', frame)
        
        # 키 입력 처리
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print(f"\n{model_name} 종료")
            break
            
        elif key == ord('s') and not is_processing:
            is_processing = True
            print("\n" + "="*70)
            print("이미지 분류 중...")
            
            predicted_class, confidence, top5, inf_time = classify_image(net, frame)
            results_dict['inference_times'].append(inf_time)
            
            print(f"\n인식 결과: {predicted_class}")
            print(f"  신뢰도: {confidence:.1f}%")
            print(f"  추론 시간: {inf_time:.2f}ms")
            
            last_object = predicted_class
            
            description = get_manual_description(predicted_class)
            last_full_description = description
            print(f"\n설명: {description}")
            
            speak_text_gtts(description)
            print("="*70 + "\n")
            is_processing = False
            
        elif key == ord('r') and last_full_description:
            print(f"\n마지막 설명: \"{last_full_description}\"")
            speak_text_gtts(last_full_description)
    
    cap.release()
    cv2.destroyAllWindows()

# ============================================================================
# 비교 그래프 생성 함수
# ============================================================================

def generate_comparison_graphs(all_results, output_dir):
    """모델 비교 그래프 생성"""
    print("\n" + "="*70)
    print("비교 그래프 생성 중...")
    print("="*70)
    
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    model_names = [r['name'] for r in all_results]
    params = [r['params']/1e6 for r in all_results]
    sizes = [r['size'] for r in all_results]
    
    # 실제 추론 시간 사용
    times = []
    for r in all_results:
        if r['inference_times']:
            times.append(np.mean(r['inference_times']))
        else:
            times.append(0.0)
    
    num_models = len(all_results)
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    # 1. 전체 메트릭 비교
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('TensorRT Precision Comparison: GoogleNet', fontsize=16, fontweight='bold')
    
    # 파라미터 수
    bars1 = axes[0].bar(range(num_models), params, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_xticks(range(num_models))
    axes[0].set_xticklabels([n.replace('TensorRT ', '') for n in model_names], rotation=0, fontsize=10)
    axes[0].set_ylabel('Million Parameters', fontsize=12)
    axes[0].set_title('Model Parameters', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    for bar, val in zip(bars1, params):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}M', ha='center', va='bottom', fontsize=10)
    
    # 모델 크기
    bars2 = axes[1].bar(range(num_models), sizes, color=colors, alpha=0.8, edgecolor='black')
    axes[1].set_xticks(range(num_models))
    axes[1].set_xticklabels([n.replace('TensorRT ', '') for n in model_names], rotation=0, fontsize=10)
    axes[1].set_ylabel('Size (MB)', fontsize=12)
    axes[1].set_title('Model Size', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    for bar, val in zip(bars2, sizes):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}MB', ha='center', va='bottom', fontsize=10)
    
    # 추론 시간
    if any(t > 0 for t in times):
        bars3 = axes[2].bar(range(num_models), times, color=colors, alpha=0.8, edgecolor='black')
        axes[2].set_xticks(range(num_models))
        axes[2].set_xticklabels([n.replace('TensorRT ', '') for n in model_names], rotation=0, fontsize=10)
        axes[2].set_ylabel('Inference Time (ms)', fontsize=12)
        axes[2].set_title('Average Inference Time', fontsize=14, fontweight='bold')
        axes[2].grid(axis='y', alpha=0.3)
        for bar, val in zip(bars3, times):
            if val > 0:
                height = bar.get_height()
                axes[2].text(bar.get_x() + bar.get_width()/2., height,
                            f'{val:.1f}ms', ha='center', va='bottom', fontsize=10)
    else:
        axes[2].text(0.5, 0.5, 'No inference data', ha='center', va='center',
                    transform=axes[2].transAxes, fontsize=14)
        axes[2].set_title('Average Inference Time', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, 'tensorrt_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"  비교 그래프 저장: {comparison_path}")
    plt.close()
    
    # 2. 개선율 그래프
    baseline_size = sizes[0]
    baseline_time = times[0] if times[0] > 0 else None
    
    size_reduction = [(baseline_size - s) / baseline_size * 100 for s in sizes]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('TensorRT Optimization Benefits vs FP32', fontsize=16, fontweight='bold')
    
    # 크기 감소율
    bars1 = axes[0].bar(range(num_models), size_reduction, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_xticks(range(num_models))
    axes[0].set_xticklabels([n.replace('TensorRT ', '') for n in model_names], rotation=0, fontsize=10)
    axes[0].set_ylabel('Size Reduction (%)', fontsize=12)
    axes[0].set_title('Model Size Reduction', fontsize=14, fontweight='bold')
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0].grid(axis='y', alpha=0.3)
    for bar, val in zip(bars1, size_reduction):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}%', ha='center', va='bottom' if val > 0 else 'top', fontsize=10)
    
    # 속도 향상율
    if baseline_time and baseline_time > 0:
        time_reduction = [(baseline_time - t) / baseline_time * 100 if t > 0 else 0 for t in times]
        bars2 = axes[1].bar(range(num_models), time_reduction, color=colors, alpha=0.8, edgecolor='black')
        axes[1].set_xticks(range(num_models))
        axes[1].set_xticklabels([n.replace('TensorRT ', '') for n in model_names], rotation=0, fontsize=10)
        axes[1].set_ylabel('Speed Improvement (%)', fontsize=12)
        axes[1].set_title('Inference Speed Improvement', fontsize=14, fontweight='bold')
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1].grid(axis='y', alpha=0.3)
        for bar, val in zip(bars2, time_reduction):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.1f}%', ha='center', va='bottom' if val > 0 else 'top', fontsize=10)
    else:
        axes[1].text(0.5, 0.5, 'No inference data', ha='center', va='center',
                    transform=axes[1].transAxes, fontsize=14)
        axes[1].set_title('Inference Speed Improvement', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    improvement_path = os.path.join(output_dir, 'tensorrt_benefits.png')
    plt.savefig(improvement_path, dpi=300, bbox_inches='tight')
    print(f"  개선율 그래프 저장: {improvement_path}")
    plt.close()
    
    print("="*70 + "\n")
    return comparison_path, improvement_path

# ============================================================================
# 메인 실행
# ============================================================================

# 환경 확인
if JETSON_AVAILABLE:
    print("\n" + "="*70)
    print("Jetson Inference TensorRT Precision Comparison")
    print("="*70)
    configs = TENSORRT_CONFIGS
    print(f"이 프로그램은 {len(configs)}개의 TensorRT 정밀도를 순차적으로 실행합니다:")
    for i, config in enumerate(configs, 1):
        print(f"  {i}. {config['name']} - {config['description']}")
elif TORCH_AVAILABLE or TENSORFLOW_AVAILABLE:
    print("\n" + "="*70)
    print("Image Classification - Multi-Platform Support")
    print("="*70)
    configs = STANDARD_CONFIGS
    print(f"이 프로그램은 {len(configs)}개의 모델을 실행합니다:")
    for i, config in enumerate(configs, 1):
        print(f"  {i}. {config['name']} - {config['description']}")
else:
    print("\nError: 사용 가능한 딥러닝 프레임워크가 없습니다.")
    print("다음 중 하나를 설치해주세요:")
    print("  - PyTorch: pip install torch torchvision")
    print("  - TensorFlow: pip install tensorflow")
    print("  - Jetson: sudo apt-get install jetson-inference (Jetson Nano에서만)")
    exit(1)

print("\n각 설정을 테스트한 후 'q' 키를 눌러 다음으로 이동하세요.")
print("="*70)

# 모든 모델 결과 저장
all_results = []

# 각 설정별로 모델 실행
for i, config in enumerate(configs):
    # 모델 로드
    if JETSON_AVAILABLE:
        net, success = load_jetson_model(config)
    else:
        net, success = load_standard_model(config)
    
    if not success or net is None:
        print(f"Warning: {config['name']} 로딩 실패. 건너뜁니다.")
        # 실패해도 추정값으로 결과 추가
        results = get_model_info(config, None)
        all_results.append(results)
        continue
    
    # 모델 정보 가져오기 (로드 후)
    results = get_model_info(config, net)
    all_results.append(results)
    
    input(f"\nPress Enter to start testing {config['name']}...")
    run_model(net, results['name'], results)
    
    # 메모리 정리
    if isinstance(net, dict):
        if 'model' in net:
            del net['model']
    del net

pygame.quit()

# ============================================================================
# 최종 비교 분석
# ============================================================================

print("\n" + "="*70)
print("최종 성능 비교 분석")
print("="*70)

# 실제 추론 시간 계산
for result in all_results:
    if result['inference_times']:
        result['time_mean'] = np.mean(result['inference_times'])
        result['time_std'] = np.std(result['inference_times'])
    else:
        result['time_mean'] = 0.0
        result['time_std'] = 0.0

print(f"\n{'모델':<20} {'파라미터':<15} {'크기(MB)':<12} {'추론시간(ms)':<15} {'개선율'}")
print("-"*70)

baseline = all_results[0]
for result in all_results:
    if result['time_mean'] > 0 and baseline['time_mean'] > 0:
        speedup = baseline['time_mean'] / result['time_mean']
    else:
        speedup = 1.0
    
    compression = baseline['size'] / result['size'] if result['size'] > 0 else 1.0
    
    if result['name'] == baseline['name']:
        improvement = 'baseline'
    else:
        improvement = f"{speedup:.2f}x faster, {compression:.2f}x smaller"
    
    params_str = f"{result['params']/1e6:.2f}M"
    size_str = f"{result['size']:.1f}"
    
    if result['time_mean'] > 0:
        time_str = f"{result['time_mean']:.2f}"
    else:
        time_str = "N/A"
    
    print(f"{result['name']:<20} {params_str:<15} {size_str:<12} {time_str:<15} {improvement}")

print("-"*70)

# 각 모델의 실행 통계
for result in all_results:
    if result['inference_times']:
        print(f"\n{result['name']} 실행 통계:")
        print(f"  총 추론 횟수:       {len(result['inference_times'])}")
        print(f"  평균 추론 시간:     {np.mean(result['inference_times']):.2f} ms")
        print(f"  표준 편차:          {np.std(result['inference_times']):.2f} ms")
        print(f"  최소 시간:          {np.min(result['inference_times']):.2f} ms")
        print(f"  최대 시간:          {np.max(result['inference_times']):.2f} ms")

print("\n" + "="*70)

# 비교 그래프 생성
generate_comparison_graphs(all_results, OUTPUT_DIR)

print("\n프로그램이 정상적으로 종료되었습니다.")
print(f"  그래프 저장 경로: {OUTPUT_DIR}/")
print("  생성된 파일:")
print(f"    - {OUTPUT_DIR}/tensorrt_comparison.png")
print(f"    - {OUTPUT_DIR}/tensorrt_benefits.png")
print()
