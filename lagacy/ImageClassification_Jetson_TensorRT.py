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

# GUR 모델 import
try:
    from GUR_model import LightweightCaptionDecoder
    GUR_MODEL_AVAILABLE = True
except ImportError:
    GUR_MODEL_AVAILABLE = False
    print("Warning: GUR_model.py를 찾을 수 없습니다. 캡션 생성 기능이 비활성화됩니다.")

# ============================================================================
# 환경 감지
# ============================================================================
PLATFORM = platform.system().lower()
IS_JETSON = os.path.exists('/etc/nv_tegra_release') or 'jetson' in platform.machine().lower()
IS_MAC = PLATFORM == 'darwin'
IS_WINDOWS = PLATFORM == 'windows'
IS_LINUX = PLATFORM == 'linux' and not IS_JETSON

print("플랫폼 감지: {} ({})".format(platform.system(), platform.machine()))
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
        print("TensorFlow 사용 가능: {}".format(tf.__version__))
    except ImportError:
        pass
    
    try:
        import torch
        import torchvision
        from torchvision import transforms, models
        TORCH_AVAILABLE = True
        print("PyTorch 사용 가능: {}".format(torch.__version__))
    except ImportError:
        pass
    
    try:
        import onnxruntime as ort
        ONNX_AVAILABLE = True
        print("ONNX Runtime 사용 가능: {}".format(ort.__version__))
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
print("그래프 저장 경로: {}\n".format(OUTPUT_DIR))

# ============================================================================
# 캡션 생성 모델 설정
# ============================================================================

# 기본 vocabulary 설정 (실제 사용 시 학습된 vocabulary 로드 필요)
DEFAULT_VOCAB = {
    '<start>': 0,
    '<end>': 1,
    '<pad>': 2,
    '<unk>': 3,
}

# 기본 reverse vocabulary (ID -> 단어)
DEFAULT_REV_VOCAB = {v: k for k, v in DEFAULT_VOCAB.items()}

# 캡션 모델 설정
CAPTION_MODEL_CONFIG = {
    'attention_dim': 256,
    'embed_dim': 256,
    'decoder_dim': 512,
    'vocab_size': 5000,  # 실제 vocabulary 크기에 맞게 조정 필요
    'encoder_dim': 1024,  # GoogleNet 출력 채널 수
    'model_path': None,  # 학습된 모델 가중치 경로 (있는 경우)
}

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
        "/var/cache/tensorrt/googlenet_{}.engine".format(precision.lower()),
        "~/.cache/tensorrt/googlenet_{}.engine".format(precision.lower()),
        "/tmp/tensorrt/googlenet_{}.engine".format(precision.lower())
    ]
    
    # 캐시 파일 찾기
    for cache_path in cache_paths:
        expanded_path = os.path.expanduser(cache_path)
        if os.path.exists(expanded_path):
            size_bytes = os.path.getsize(expanded_path)
            size_mb = size_bytes / (1024 * 1024)
            return size_mb
    
    # 캐시 파일이 없으면 추정값 반환
    print("  Warning: 캐시 파일을 찾을 수 없어 추정값 사용")
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
    print("{config['name']} 모델 로딩...")
    print("설명: {config['description']}")
    print("="*70)
    
    try:
        # TensorRT 엔진 빌드 (처음 실행 시 시간 소요)
        print("\nTensorRT 엔진 빌드 중 ({config['precision']})...")
        print("주의: 처음 실행 시 5-10분 소요될 수 있습니다.")
        
        # Network 파라미터 구성
        argv = ['--network={config["network"]}']
        
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
        print("Error: TensorRT 모델 로딩 실패 - {}".format(e))
        return None, False

def load_pytorch_model(config):
    """PyTorch 모델 로드 (Mac/Windows용)"""
    if not TORCH_AVAILABLE:
        return None, False
    
    print("\n" + "="*70)
    print("{config['name']} 모델 로딩...")
    print("설명: {config['description']}")
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
        
        # 특징 맵 추출을 위한 extractor 생성
        extractor = FeatureMapExtractor()
        
        print("완료!\n")
        return {'model': model, 'transform': transform, 'extractor': extractor}, True
        
    except Exception as e:
        print("Error: PyTorch 모델 로딩 실패 - {}".format(e))
        return None, False

def load_tensorflow_model(config):
    """TensorFlow 모델 로드 (Mac/Windows용)"""
    if not TENSORFLOW_AVAILABLE:
        return None, False
    
    print("\n" + "="*70)
    print("{config['name']} 모델 로딩...")
    print("설명: {config['description']}")
    print("="*70)
    
    try:
        print("\nTensorFlow 모델 로딩 중...")
        # 먼저 Keras 애플리케이션 시도 (더 안정적)
        try:
            from tensorflow.keras.applications import InceptionV3
            from tensorflow.keras.applications.inception_v3 import preprocess_input
            
            base_model = InceptionV3(weights='imagenet')
            
            # 특징 맵 추출을 위한 중간 레이어 모델 생성
            # 마지막 conv 레이어 직전까지의 모델
            feature_model = None
            try:
                # InceptionV3의 마지막 conv 블록 (mixed10) 출력 추출
                layer_name = 'mixed10'
                if hasattr(base_model, 'get_layer'):
                    feature_layer = base_model.get_layer(layer_name)
                    feature_model = tf.keras.Model(
                        inputs=base_model.input,
                        outputs=feature_layer.output
                    )
                    print("  특징 맵 추출 레이어: {}".format(layer_name))
            except Exception as e:
                print("  Warning: 특징 맵 모델 생성 실패 - {}".format(e))
            
            print("완료! (Keras InceptionV3 사용)\n")
            return {
                'model': base_model, 
                'preprocess': preprocess_input,
                'feature_model': feature_model
            }, True
        except Exception as e1:
            # 대체: TensorFlow Hub에서 모델 로드
            try:
                import tensorflow_hub as hub
                model_url = "https://tfhub.dev/google/imagenet/inception_v3/classification/5"
                model = hub.load(model_url)
                print("완료! (TensorFlow Hub 사용)\n")
                # Hub 모델은 특징 맵 추출이 제한적
                return {'model': model, 'feature_model': None}, True
            except Exception as e2:
                print("Error: TensorFlow Hub 모델 로딩 실패 - {}".format(e2))
                raise e1
        
    except Exception as e:
        print("Error: TensorFlow 모델 로딩 실패 - {}".format(e))
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
    
    print("\n{'='*70}")
    print("{config['name']} 모델 사양 (GoogleNet)")
    print("{'='*70}")
    if 'network' in config:
        print("  네트워크:           {config['network']}")
    if 'framework' in config:
        print("  프레임워크:         {config['framework']}")
    print("  정밀도:             {}".format(precision))
    print("  파라미터 수:        {} ({}M)".format(params:,, params/1e6:.2f))
    print("  모델 크기:          {} MB".format(size:.2f))
    print("  FLOPs:              {} ({}G)".format(flops:,, flops/1e9:.2f))
    print("{'='*70}\n")
    
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
# 특징 맵 추출 유틸리티
# ============================================================================

class FeatureMapExtractor:
    """특징 맵 추출을 위한 클래스"""
    def __init__(self):
        self.feature_maps = {}
        self.hooks = []
    
    def register_hook(self, name, module):
        """PyTorch forward hook 등록"""
        def hook_fn(module, input, output):
            # GPU에서 CPU로 이동하고 numpy로 변환
            if isinstance(output, torch.Tensor):
                self.feature_maps[name] = output.detach().cpu().numpy()
            else:
                self.feature_maps[name] = output
        
        hook = module.register_forward_hook(hook_fn)
        self.hooks.append(hook)
        return hook
    
    def clear_hooks(self):
        """등록된 hook 제거"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.feature_maps = {}
    
    def get_feature_map(self, layer_name):
        """특정 레이어의 특징 맵 반환"""
        return self.feature_maps.get(layer_name, None)

# ============================================================================
# 분류 함수
# ============================================================================

def classify_image_jetson(net, frame, extract_features=False):
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
    
    # 특징 맵 추출 (Jetson에서는 제한적)
    feature_maps = None
    if extract_features:
        # Jetson API는 중간 레이어 접근이 제한적
        feature_maps = {
            'layer_name': 'output',
            'feature_map': None,
            'shape': None,
            'available': False,
            'message': 'Jetson TensorRT API는 중간 레이어 접근을 지원하지 않습니다.'
        }
    
    return predicted_class, confidence_percent, top5_results, inference_time, feature_maps

def classify_image_pytorch(model_dict, frame, extract_features=False):
    """PyTorch용 이미지 분류"""
    model = model_dict['model']
    transform = model_dict['transform']
    extractor = model_dict.get('extractor', None)
    
    # OpenCV BGR을 RGB로 변환
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 전처리
    input_tensor = transform(rgb_frame)
    input_batch = input_tensor.unsqueeze(0)
    
    # GPU 사용 가능 시
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_batch = input_batch.to(device)
    
    # 특징 맵 추출을 위한 hook 설정
    feature_maps = None
    if extract_features and extractor is not None:
        extractor.clear_hooks()
        # GoogleNet의 마지막 Inception 모듈 (inception5b)에서 특징 맵 추출
        if hasattr(model, 'inception5b'):
            extractor.register_hook('inception5b', model.inception5b)
        elif hasattr(model, 'Inception5b'):
            extractor.register_hook('inception5b', model.Inception5b)
        else:
            # 대체: 마지막 conv 레이어 찾기
            for name, module in model.named_modules():
                if 'conv' in name.lower() and len(list(module.children())) == 0:
                    extractor.register_hook(name, module)
                    break
    
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
    
    # 특징 맵 추출
    if extract_features and extractor is not None:
        feature_map_data = None
        layer_name = None
        for name, fm in extractor.feature_maps.items():
            feature_map_data = fm
            layer_name = name
            break
        
        if feature_map_data is not None:
            # 배치 차원 제거 (배치 크기가 1인 경우)
            if len(feature_map_data.shape) == 4 and feature_map_data.shape[0] == 1:
                feature_map_data = feature_map_data[0]
            
            feature_maps = {
                'layer_name': layer_name or 'unknown',
                'feature_map': feature_map_data,
                'shape': feature_map_data.shape if feature_map_data is not None else None,
                'available': True
            }
        else:
            feature_maps = {
                'layer_name': 'unknown',
                'feature_map': None,
                'shape': None,
                'available': False,
                'message': '특징 맵을 추출할 수 없습니다.'
            }
    
    return predicted_class, confidence_percent, top5_results, inference_time, feature_maps

def classify_image_tensorflow(model_dict, frame, extract_features=False):
    """TensorFlow용 이미지 분류"""
    model = model_dict['model']
    preprocess = model_dict.get('preprocess', None)
    feature_model = model_dict.get('feature_model', None)
    
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
    
    # 특징 맵 추출
    feature_maps = None
    if extract_features:
        if feature_model is not None:
            try:
                feature_map_data = feature_model(img_array)
                if hasattr(feature_map_data, 'numpy'):
                    feature_map_data = feature_map_data.numpy()
                
                # 배치 차원 제거
                if len(feature_map_data.shape) == 4 and feature_map_data.shape[0] == 1:
                    feature_map_data = feature_map_data[0]
                
                feature_maps = {
                    'layer_name': 'intermediate_layer',
                    'feature_map': feature_map_data,
                    'shape': feature_map_data.shape,
                    'available': True
                }
            except Exception as e:
                feature_maps = {
                    'layer_name': 'unknown',
                    'feature_map': None,
                    'shape': None,
                    'available': False,
                    'message': '특징 맵 추출 실패: {}' .format(str(e))
                }
        else:
            feature_maps = {
                'layer_name': 'unknown',
                'feature_map': None,
                'shape': None,
                'available': False,
                'message': '특징 맵 모델이 설정되지 않았습니다.'
            }
    
    return predicted_class, confidence_percent, top5_results, inference_time, feature_maps

def classify_image(net, frame, extract_features=False):
    """통합 이미지 분류 함수 (환경 자동 감지)"""
    if JETSON_AVAILABLE and isinstance(net, jetson.inference.imageNet):
        return classify_image_jetson(net, frame, extract_features)
    elif isinstance(net, dict):
        if 'transform' in net:  # PyTorch
            return classify_image_pytorch(net, frame, extract_features)
        elif 'model' in net:  # TensorFlow
            return classify_image_tensorflow(net, frame, extract_features)
    
    # 기본값
    print("Error: 알 수 없는 모델 타입")
    return "unknown", 0.0, [], 0.0, None

def get_manual_description(object_name):
    """수동 설명 반환"""
    object_lower = object_name.lower()
    
    if object_lower in MANUAL_DESCRIPTIONS:
        return MANUAL_DESCRIPTIONS[object_lower]
    
    for key in MANUAL_DESCRIPTIONS.keys():
        if key in object_lower or object_lower in key:
            return MANUAL_DESCRIPTIONS[key]
    
    return "This is a {}.".format(object_name)

# ============================================================================
# 특징 맵 저장 및 시각화 함수
# ============================================================================

def save_feature_map(feature_maps, model_name, predicted_class, output_dir):
    """특징 맵 저장"""
    if feature_maps is None or not feature_maps.get('available', False):
        print("  Warning: 특징 맵을 저장할 수 없습니다.")
        return None
    
    feature_map_dir = os.path.join(output_dir, 'feature_maps', model_name.replace(' ', '_'))
    os.makedirs(feature_map_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_filename = "feature_map_{}_{}".format(predicted_class, timestamp)
    
    # NumPy 배열 저장
    feature_map_data = feature_maps['feature_map']
    npy_path = os.path.join(feature_map_dir, "{}.npy".format(base_filename))
    np.save(npy_path, feature_map_data)
    print("  특징 맵 저장: {}".format(npy_path))
    
    # 메타데이터 저장
    metadata = {
        'layer_name': feature_maps.get('layer_name', 'unknown'),
        'shape': feature_maps.get('shape', None),
        'predicted_class': predicted_class,
        'timestamp': timestamp,
        'model_name': model_name
    }
    metadata_path = os.path.join(feature_map_dir, "{}_metadata.json".format(base_filename))
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return npy_path

def visualize_feature_map(feature_maps, save_path=None, top_k=16):
    """특징 맵 시각화"""
    if feature_maps is None or not feature_maps.get('available', False):
        print("  Warning: 특징 맵을 시각화할 수 없습니다.")
        return None
    
    feature_map_data = feature_maps['feature_map']
    
    # 특징 맵 shape 확인 및 변환
    if len(feature_map_data.shape) == 3:
        # Shape 판단: 첫 번째 차원이 가장 크면 (C, H, W), 마지막 차원이 가장 크면 (H, W, C)
        if feature_map_data.shape[0] > feature_map_data.shape[2]:
            # (C, H, W) 형식
            channels, height, width = feature_map_data.shape
            # 각 채널을 개별적으로 처리하기 위해 (H, W, C)로 변환하지 않고 그대로 사용
            channel_data = feature_map_data  # (C, H, W) 유지
        else:
            # (H, W, C) 형식
            height, width, channels = feature_map_data.shape
            # (C, H, W)로 변환
            channel_data = np.transpose(feature_map_data, (2, 0, 1))
    else:
        print("  Warning: 예상치 못한 특징 맵 shape: {}".format(feature_map_data.shape))
        return None
    
    print("  특징 맵 shape: ({}, {}, {})".format(channels, height, width))
    
    # 채널별로 정규화 및 시각화
    num_channels = min(channels, top_k)
    
    # 각 채널의 활성화 값 계산 (평균 및 최대값)
    channel_activations = []
    for i in range(channels):
        channel = channel_data[i]  # (H, W)
        mean_activation = np.mean(channel)
        max_activation = np.max(channel)
        std_activation = np.std(channel)
        # 활성화 점수: 평균 + 표준편차 (더 활성화된 채널 우선)
        activation_score = mean_activation + 0.5 * std_activation
        channel_activations.append((i, mean_activation, max_activation, activation_score))
    
    # 상위 활성화 채널 선택 (activation_score 기준)
    channel_activations.sort(key=lambda x: x[3], reverse=True)
    top_channels = [idx for idx, _, _, _ in channel_activations[:num_channels]]
    
    # 그리드로 시각화
    cols = 4
    rows = (num_channels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    fig.suptitle('Feature Maps - Layer: {feature_maps.get("layer_name", "unknown")} (Top {num_channels}/{channels} channels)', 
                 fontsize=16, fontweight='bold')
    
    # axes 배열 정규화
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1) if hasattr(axes, 'reshape') else np.array([axes])
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else axes
        axes = axes.reshape(rows, cols)
    
    for idx, channel_idx in enumerate(top_channels):
        row = idx // cols
        col = idx % cols
        
        channel = channel_data[channel_idx]  # (H, W)
        
        # 정규화 (0-1 범위)
        channel_min = channel.min()
        channel_max = channel.max()
        if channel_max > channel_min:
            channel_norm = (channel - channel_min) / (channel_max - channel_min)
        else:
            channel_norm = np.zeros_like(channel)
        
        # 시각화 (더 나은 컬러맵 사용)
        im = axes[row, col].imshow(channel_norm, cmap='hot', interpolation='bilinear', aspect='auto')
        
        # 제목에 활성화 정보 표시
        mean_val = channel_activations[idx][1]
        max_val = channel_activations[idx][2]
        axes[row, col].set_title('Channel {}\nMean: {} | Max: {}' .format(channel_idx, mean_val:.3f, max_val:.3f), 
                                fontsize=9, pad=5)
        axes[row, col].axis('of')
    
    # 빈 subplot 숨기기
    for idx in range(num_channels, rows * cols):
        row = idx // cols
        col = idx % cols
        if isinstance(axes, np.ndarray) and axes.ndim == 2:
            axes[row, col].axis('of')
        else:
            try:
                axes[idx].axis('of')
            except:
                pass
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print("  특징 맵 시각화 저장: {}".format(save_path))
    else:
        plt.show()
    
    plt.close()
    
    return save_path

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
# 캡션 생성 함수
# ============================================================================

def load_caption_model(config=None):
    """캡션 생성 모델 로드"""
    if not GUR_MODEL_AVAILABLE or not TORCH_AVAILABLE:
        return None
    
    if config is None:
        config = CAPTION_MODEL_CONFIG
    
    try:
        print("\n캡션 생성 모델 초기화 중...")
        decoder = LightweightCaptionDecoder(
            attention_dim=config['attention_dim'],
            embed_dim=config['embed_dim'],
            decoder_dim=config['decoder_dim'],
            vocab_size=config['vocab_size'],
            encoder_dim=config['encoder_dim']
        )
        
        # 학습된 가중치 로드 (있는 경우)
        if config.get('model_path') and os.path.exists(config['model_path']):
            decoder.load_state_dict(torch.load(config['model_path'], map_location='cpu'))
            print("  학습된 가중치 로드: {config['model_path']}")
        else:
            print("  Warning: 학습된 가중치가 없습니다. 랜덤 초기화된 모델을 사용합니다.")
            print("  실제 사용 시 학습된 모델을 로드해야 합니다.")
        
        decoder.eval()
        print("  완료!\n")
        return decoder
        
    except Exception as e:
        print("Error: 캡션 모델 로드 실패 - {}".format(e))
        return None

def generate_caption_from_feature_map(feature_maps, caption_model, vocab=None, rev_vocab=None):
    """특징 맵으로부터 캡션 생성"""
    if not GUR_MODEL_AVAILABLE or not TORCH_AVAILABLE:
        return None, "PyTorch 또는 GUR 모델을 사용할 수 없습니다."
    
    if caption_model is None:
        return None, "캡션 모델이 로드되지 않았습니다."
    
    if feature_maps is None or not feature_maps.get('available', False):
        return None, "특징 맵을 사용할 수 없습니다."
    
    try:
        feature_map_data = feature_maps['feature_map']
        
        # NumPy 배열을 PyTorch 텐서로 변환
        if isinstance(feature_map_data, np.ndarray):
            # Shape 확인 및 변환: (C, H, W) 형식으로 변환
            if len(feature_map_data.shape) == 3:
                if feature_map_data.shape[0] > feature_map_data.shape[2]:
                    # 이미 (C, H, W) 형식
                    feature_tensor = torch.from_numpy(feature_map_data).float()
                else:
                    # (H, W, C) 형식 -> (C, H, W)로 변환
                    feature_tensor = torch.from_numpy(np.transpose(feature_map_data, (2, 0, 1))).float()
            else:
                return None, "예상치 못한 특징 맵 shape: {}".format(feature_map_data.shape)
        else:
            feature_tensor = feature_map_data
        
        # GPU 사용 가능 시
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        caption_model = caption_model.to(device)
        feature_tensor = feature_tensor.to(device)
        
        # 캡션 생성
        caption_ids, attention_weights = caption_model.generate_caption(
            feature_tensor,
            max_len=20,
            start_token=0,
            end_token=1
        )
        
        # ID를 단어로 변환
        if rev_vocab is None:
            rev_vocab = DEFAULT_REV_VOCAB
        
        caption_words = []
        for word_id in caption_ids:
            if word_id in rev_vocab:
                word = rev_vocab[word_id]
                if word not in ['<start>', '<end>', '<pad>', '<unk>']:
                    caption_words.append(word)
            else:
                caption_words.append("<{}>".format(word_id))
        
        caption_text = " ".join(caption_words) if caption_words else "No caption generated"
        
        return caption_text, attention_weights
        
    except Exception as e:
        return None, "캡션 생성 실패: {}".format(str(e))

# ============================================================================
# 모델 실행 함수
# ============================================================================

def run_model(net, model_name, results_dict):
    """모델 실행"""
    if net is None:
        print("Error: {} 모델을 사용할 수 없습니다.".format(model_name))
        return
    
    # 캡션 모델 로드 (가능한 경우)
    caption_model = None
    if GUR_MODEL_AVAILABLE and TORCH_AVAILABLE:
        caption_model = load_caption_model()
    
    cap = cv2.VideoCapture(0)
    
    print("\n" + "="*70)
    print("=== {} 실행 중 ===".format(model_name))
    print("="*70)
    print("\nKey commands:")
    print("  's' : 분류 및 음성 출력 (특징 맵 자동 추출)")
    print("  '' : 특징 맵 저장 및 시각화")
    print("  'v' : 특징 맵 시각화 창 표시")
    print("  'c' : 캡션 생성 (특징 맵 사용)")
    print("  'r' : 마지막 설명 다시 듣기")
    print("  'q' : 다음 모델로 이동\n")
    
    description = ""
    last_object = None
    last_full_description = ""
    last_feature_maps = None
    last_caption = None
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
        cv2.putText(frame, "Model: {}".format(model_name), (10, frame.shape[0] - 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # 평균 추론 시간 표시
        if results_dict['inference_times']:
            avg_inf_time = np.mean(results_dict['inference_times'][-30:])
            cv2.rectangle(frame, (frame.shape[1] - 200, frame.shape[0] - 35), 
                         (frame.shape[1] - 5, frame.shape[0] - 5), (50, 50, 50), -1)
            cv2.putText(frame, "Inf: {}ms".format(avg_inf_time:.1f), (frame.shape[1] - 190, frame.shape[0] - 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 현재 인식된 객체 표시
        if last_object and not is_processing:
            cv2.rectangle(frame, (5, 5), (450, 55), (0, 0, 0), -1)
            cv2.putText(frame, "Detected: {}".format(last_object), (10, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # 생성된 캡션 표시
        if last_caption and not is_processing:
            caption_y = 60
            max_width = frame.shape[1] - 20
            words = last_caption.split()
            line = ""
            line_num = 0
            
            for word in words:
                test_line = line + word + " "
                text_size = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                
                if text_size[0] > max_width:
                    text_size_actual = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                    cv2.rectangle(frame, (5, caption_y + line_num * 20 - 15), 
                                (15 + text_size_actual[0], caption_y + line_num * 20 + 5), 
                                (0, 0, 0), -1)
                    cv2.putText(frame, line, (10, caption_y + line_num * 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)
                    line = word + " "
                    line_num += 1
                else:
                    line = test_line
            
            if line:
                text_size_actual = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                cv2.rectangle(frame, (5, caption_y + line_num * 20 - 15), 
                            (15 + text_size_actual[0], caption_y + line_num * 20 + 5), 
                            (0, 0, 0), -1)
                cv2.putText(frame, line, (10, caption_y + line_num * 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)
        
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
        
        cv2.imshow('{}' .format(model_name), frame)
        
        # 키 입력 처리
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n{} 종료".format(model_name))
            break
            
        elif key == ord('s') and not is_processing:
            is_processing = True
            print("\n" + "="*70)
            print("이미지 분류 중...")
            
            # 특징 맵도 함께 추출
            predicted_class, confidence, top5, inf_time, feature_maps = classify_image(
                net, frame, extract_features=True
            )
            results_dict['inference_times'].append(inf_time)
            
            print("\n인식 결과: {}".format(predicted_class))
            print("  신뢰도: {}%".format(confidence:.1f))
            print("  추론 시간: {}ms".format(inf_time:.2f))
            
            # 특징 맵 정보 출력
            if feature_maps and feature_maps.get('available', False):
                print("  특징 맵: {feature_maps.get('layer_name', 'unknown')}")
                print("    Shape: {feature_maps.get('shape', 'unknown')}")
            elif feature_maps:
                print("  특징 맵: 추출 불가 - {feature_maps.get('message', 'unknown error')}")
            
            last_object = predicted_class
            last_feature_maps = feature_maps if feature_maps and feature_maps.get('available', False) else None
            
            description = get_manual_description(predicted_class)
            last_full_description = description
            print("\n설명: {}".format(description))
            
            speak_text_gtts(description)
            print("="*70 + "\n")
            is_processing = False
            
        elif key == ord('') and not is_processing and last_feature_maps is not None:
            # 특징 맵 저장 및 시각화
            print("\n" + "="*70)
            print("특징 맵 저장 및 시각화 중...")
            
            npy_path = save_feature_map(last_feature_maps, model_name, last_object, OUTPUT_DIR)
            if npy_path:
                # 시각화 저장 경로
                viz_path = npy_path.replace('.npy', '_visualization.png')
                visualize_feature_map(last_feature_maps, save_path=viz_path)
            
            print("="*70 + "\n")
            
        elif key == ord('v') and not is_processing and last_feature_maps is not None:
            # 특징 맵 시각화 창 표시
            print("\n특징 맵 시각화 중...")
            visualize_feature_map(last_feature_maps, save_path=None)
            
        elif key == ord('c') and not is_processing and last_feature_maps is not None:
            # 캡션 생성
            is_processing = True
            print("\n" + "="*70)
            print("캡션 생성 중...")
            
            if caption_model is None:
                print("  Error: 캡션 모델이 로드되지 않았습니다.")
                print("  GUR_model.py와 PyTorch가 필요합니다.")
            else:
                caption_text, attention_weights = generate_caption_from_feature_map(
                    last_feature_maps, caption_model
                )
                
                if caption_text:
                    last_caption = caption_text
                    print("\n생성된 캡션: {}".format(caption_text))
                    print("  어텐션 맵 수: {}".format(len(attention_weights) if attention_weights else 0))
                    
                    # 캡션 음성 출력
                    speak_text_gtts(caption_text)
                else:
                    print("  Error: {attention_weights if isinstance(attention_weights, str) else '캡션 생성 실패'}")
            
            print("="*70 + "\n")
            is_processing = False
            
        elif key == ord('r') and last_full_description:
            print("\n마지막 설명: \"{}\"".format(last_full_description))
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
                    '{}M' .format(val:.2f), ha='center', va='bottom', fontsize=10)
    
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
                    '{}MB' .format(val:.1f), ha='center', va='bottom', fontsize=10)
    
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
                            '{}ms' .format(val:.1f), ha='center', va='bottom', fontsize=10)
    else:
        axes[2].text(0.5, 0.5, 'No inference data', ha='center', va='center',
                    transform=axes[2].transAxes, fontsize=14)
        axes[2].set_title('Average Inference Time', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, 'tensorrt_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print("  비교 그래프 저장: {}".format(comparison_path))
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
                    '{}%' .format(val:.1f), ha='center', va='bottom' if val > 0 else 'top', fontsize=10)
    
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
                        '{}%' .format(val:.1f), ha='center', va='bottom' if val > 0 else 'top', fontsize=10)
    else:
        axes[1].text(0.5, 0.5, 'No inference data', ha='center', va='center',
                    transform=axes[1].transAxes, fontsize=14)
        axes[1].set_title('Inference Speed Improvement', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    improvement_path = os.path.join(output_dir, 'tensorrt_benefits.png')
    plt.savefig(improvement_path, dpi=300, bbox_inches='tight')
    print("  개선율 그래프 저장: {}".format(improvement_path))
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
    print("이 프로그램은 {}개의 TensorRT 정밀도를 순차적으로 실행합니다:".format(len(configs)))
    for i, config in enumerate(configs, 1):
        print("  {}. {config[".format(i)name']} - {config['description']}")
elif TORCH_AVAILABLE or TENSORFLOW_AVAILABLE:
    print("\n" + "="*70)
    print("Image Classification - Multi-Platform Support")
    print("="*70)
    configs = STANDARD_CONFIGS
    print("이 프로그램은 {}개의 모델을 실행합니다:".format(len(configs)))
    for i, config in enumerate(configs, 1):
        print("  {}. {config[".format(i)name']} - {config['description']}")
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
        print("Warning: {config['name']} 로딩 실패. 건너뜁니다.")
        # 실패해도 추정값으로 결과 추가
        results = get_model_info(config, None)
        all_results.append(results)
        continue
    
    # 모델 정보 가져오기 (로드 후)
    results = get_model_info(config, net)
    all_results.append(results)
    
    input("\nPress Enter to start testing {config['name']}...")
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

print("\n{'모델':<20} {'파라미터':<15} {'크기(MB)':<12} {'추론시간(ms)':<15} {'개선율'}")
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
        improvement = "{}x faster, {}x smaller".format(speedup:.2f, compression:.2f)
    
    params_str = "{result['params']/1e6:.2f}M"
    size_str = "{result['size']:.1f}"
    
    if result['time_mean'] > 0:
        time_str = "{result['time_mean']:.2f}"
    else:
        time_str = "N/A"
    
    print("{result['name']:<20} {params_str:<15} {size_str:<12} {time_str:<15} {improvement}")

print("-"*70)

# 각 모델의 실행 통계
for result in all_results:
    if result['inference_times']:
        print("\n{result['name']} 실행 통계:")
        print("  총 추론 횟수:       {len(result['inference_times'])}")
        print("  평균 추론 시간:     {np.mean(result['inference_times']):.2f} ms")
        print("  표준 편차:          {np.std(result['inference_times']):.2f} ms")
        print("  최소 시간:          {np.min(result['inference_times']):.2f} ms")
        print("  최대 시간:          {np.max(result['inference_times']):.2f} ms")

print("\n" + "="*70)

# 비교 그래프 생성
generate_comparison_graphs(all_results, OUTPUT_DIR)

print("\n프로그램이 정상적으로 종료되었습니다.")
print("  그래프 저장 경로: {}/".format(OUTPUT_DIR))
print("  생성된 파일:")
print("    - {}/tensorrt_comparison.png".format(OUTPUT_DIR))
print("    - {}/tensorrt_benefits.png".format(OUTPUT_DIR))
print()
