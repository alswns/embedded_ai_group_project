"""
공통 설정 및 유틸리티
"""
import torch
import matplotlib
import matplotlib.pyplot as plt
import platform
from torchvision import transforms

# 기본 경로 설정
MODEL_PATH = "models/lightweight_captioning_model.pth"
TEST_IMAGE_DIR = "assets/images"
CAPTIONS_FILE = "assets/captions.txt"

def setup_device():
    """디바이스 선택 (CUDA > MPS > CPU)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"🚀 실행 디바이스: {device}")
    return device

def setup_matplotlib():
    """Matplotlib 한글 폰트 설정"""
    matplotlib.use('Agg')  # GUI 없이 백그라운드에서 실행
    
    os_name = platform.system()
    if os_name == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
    elif os_name == 'Darwin':  # macOS
        plt.rcParams['font.family'] = 'AppleGothic'
        plt.rcParams['axes.unicode_minus'] = False
    elif os_name == 'Linux':
        plt.rcParams['font.family'] = 'NanumGothic'
        plt.rcParams['axes.unicode_minus'] = False
    else:
        plt.rcParams['axes.unicode_minus'] = False

def get_image_transform():
    """이미지 전처리 transform 반환"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

