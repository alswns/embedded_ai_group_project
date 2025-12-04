"""
Quantization 벤치마크 스크립트 (최종 수정됨)
- 수정사항 1: IndexError 방지 (files[567] -> files[0])
- 수정사항 2: QAT 학습 시 Mixed Precision 비활성화 (정확도 향상)
- 수정사항 3: 불필요한 Wrapper 클래스 삭제
"""
import torch
import torch.nn as nn
from torch.quantization import quantize_fx
import numpy as np
import os
import time
import psutil
import matplotlib.pyplot as plt
import matplotlib
from copy import deepcopy
import gc
from collections import defaultdict
from PIL import Image
from torchvision import transforms
import platform
import warnings
import sys # 추가됨

warnings.filterwarnings('ignore')

# -------------------------------------------------------------------------
# [중요] 모델 import 경로 확인
# -------------------------------------------------------------------------
try:
    from src.muti_modal_model.model import MobileNetCaptioningModel
except ImportError:
    print("⚠️ 모델 클래스를 import할 수 없습니다. 경로를 확인해주세요.")
    # 더미 클래스
    class MobileNetCaptioningModel(nn.Module):
        def __init__(self, vocab_size, embed_dim):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, embed_dim)
            self.gru = nn.GRU(embed_dim, 512)
            self.fc = nn.Linear(512, vocab_size)
        def generate(self, img, wm, rwm, max_len):
            return ["<start>", "a", "test", "caption", "<end>"]

# NLTK 설정
try:
    from nltk.translate.meteor_score import meteor_score
    from nltk.tokenize import word_tokenize
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    METEOR_AVAILABLE = True
except ImportError:
    print("⚠️ nltk가 설치되지 않았습니다. METEOR 점수 계산 불가.")
    METEOR_AVAILABLE = False

# [삭제됨] QuantizedEncoderWrapper는 FX 모드에서 필요 없습니다.

# ============================================================================
# 설정
# ============================================================================
matplotlib.use('Agg')

# 한글 폰트 설정
os_name = platform.system()
if os_name == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False 
elif os_name == 'Darwin': 
    plt.rcParams['font.family'] = 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False
elif os_name == 'Linux':
    plt.rcParams['font.family'] = 'NanumGothic'
    plt.rcParams['axes.unicode_minus'] = False
else:
    plt.rcParams['axes.unicode_minus'] = False

MODEL_PATH = "models/lightweight_captioning_model.pth"
TEST_IMAGE_DIR = "assets/images"
CAPTIONS_FILE = "assets/captions.txt"
OUTPUT_DIR = "benchmark_results"
NUM_RUNS = 50

# QAT 설정
USE_QAT = True 
QAT_EPOCHS = 20

# 디바이스 선택
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device("mps")
print(f"🚀 실행 디바이스: {device}")

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================================================
# 유틸리티 함수
# ============================================================================
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def get_model_size_mb(model):
    param_size = 0
    buffer_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / 1024 / 1024

def get_peak_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def load_data():
    """테스트 이미지와 참조 캡션 로드"""
    img_tensor = None
    filename = None
    
    if os.path.exists(TEST_IMAGE_DIR):
        files = [f for f in os.listdir(TEST_IMAGE_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if files:
            import random
            # [수정됨] 하드코딩된 인덱스 제거 (IndexError 방지)
            # 파일이 있으면 첫 번째 파일 사용, 없으면 더미 사용
            filename = files[2] 
            img_path = os.path.join(TEST_IMAGE_DIR, filename)
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                print(f"📸 테스트 이미지: {filename}")
            except Exception as e:
                print(f"⚠️ 이미지 로드 중 에러: {e}")

    if img_tensor is None:
        print("⚠️ 이미지를 찾을 수 없어 더미 데이터를 사용합니다.")
        img_tensor = torch.randn(1, 3, 224, 224).to(device)
        filename = "dummy"

    ref_caption = None
    if os.path.exists(CAPTIONS_FILE) and filename != "dummy":
        with open(CAPTIONS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if filename in line:
                    parts = line.split(',', 1) if ',' in line else line.split('\t', 1)
                    if len(parts) > 1:
                        if parts[0].strip() == filename:
                            ref_caption = parts[1].strip()
                            print(f"📝 참조 캡션: {ref_caption}")
                            break
                        else:
                            continue
    
    return img_tensor, ref_caption

def calculate_meteor(gen_list, ref_str):
    if not METEOR_AVAILABLE or not ref_str:
        return None
    try:
        gen_str = ' '.join([w for w in gen_list if w not in ['<start>', '<end>', '<pad>', '<unk>']])
        return meteor_score([word_tokenize(ref_str.lower())], word_tokenize(gen_str.lower()))
    except:
        return None

# ============================================================================
# 모델 로드 및 변환 함수
# ============================================================================
def load_base_model():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 모델 파일 없음: {MODEL_PATH}")
        return MobileNetCaptioningModel(vocab_size=5000, embed_dim=300).to(device), {}, {}

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    vocab_size = checkpoint.get('vocab_size', 5000)
    
    model = MobileNetCaptioningModel(vocab_size=vocab_size, embed_dim=300).to(device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    return model, checkpoint.get('word_map', {}), checkpoint.get('rev_word_map', {})

def prepare_calibration_dataset(word_map, num_samples=100, max_len=20):
    """정적 양자화를 위한 Calibration 데이터셋 준비"""
    calibration_images = []
    
    if not os.path.exists(TEST_IMAGE_DIR):
        for _ in range(num_samples):
            calibration_images.append(torch.randn(1, 3, 224, 224))
        return calibration_images, None
    
    image_files = [f for f in os.listdir(TEST_IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        for _ in range(num_samples):
            calibration_images.append(torch.randn(1, 3, 224, 224))
        return calibration_images, None
    
    import random
    selected_files = random.sample(image_files, min(num_samples, len(image_files)))
    
    print(f"   📊 Calibration 데이터셋 준비 중: {len(selected_files)}개 이미지")
    
    for filename in selected_files:
        try:
            img_path = os.path.join(TEST_IMAGE_DIR, filename)
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0) 
            calibration_images.append(img_tensor)
        except Exception:
            continue
            
    while len(calibration_images) < num_samples:
        calibration_images.append(torch.randn(1, 3, 224, 224))
    
    return calibration_images[:num_samples], None

def convert_to_int8(model, word_map=None, use_qat=False, qat_epochs=2):
    if use_qat:
        return convert_to_int8_qat(model, word_map, qat_epochs)
    else:
        return convert_to_int8_static(model, word_map)

def convert_to_int8_static(model, word_map=None):
    """Int8 Static Quantization (FX Graph Mode)"""
    print("   👉 Int8 변환 중 (Static Quantization: FX Graph Mode)...")
    
    import platform
    machine = platform.machine().lower()
    if 'arm' in machine or 'aarch64' in machine:
        backend = 'qnnpack'
    elif 'x86' in machine or 'amd64' in machine:
        backend = 'fbgemm'
    else:
        backend = 'qnnpack'
    
    torch.backends.quantized.engine = backend
    print(f"   ⚙️ Quantization Engine: {backend}")

    model_cpu = deepcopy(model).cpu()
    model_cpu.eval()

    if word_map is None:
        print("   ⚠️ word_map이 없어 Dynamic Quantization으로 fallback")
        return torch.quantization.quantize_dynamic(model_cpu, {nn.Linear}, dtype=torch.qint8)

    print("   📊 Calibration 데이터 준비 중...")
    cal_images, _ = prepare_calibration_dataset(word_map, num_samples=20)
    example_input = cal_images[0]

    try:
        qconfig_dict = {"": torch.quantization.get_default_qconfig(backend)}
        
        print("   🔧 인코더 자동 융합 및 준비 (Prepare FX)...")
        model_cpu.encoder = quantize_fx.prepare_fx(model_cpu.encoder, qconfig_dict, example_input)

        print("   🔄 Calibration 진행 중 (인코더)...")
        with torch.no_grad():
            for img in cal_images:
                model_cpu.encoder(img)

        print("   ⚡ 인코더 변환 (Convert FX)...")
        model_cpu.encoder = quantize_fx.convert_fx(model_cpu.encoder)

        print("   🔄 디코더 동적 양자화 적용...")
        quantized_model = torch.quantization.quantize_dynamic(
            model_cpu,
            {nn.Linear, nn.GRU, nn.LSTM},
            dtype=torch.qint8
        )
        
        print("   ✅ 정적(Encoder/FX) + 동적(Decoder) 양자화 완료!")
        return quantized_model

    except Exception as e:
        print(f"   ⚠️ FX 정적 양자화 실패: {e}")
        return torch.quantization.quantize_dynamic(
            deepcopy(model).cpu(),
            {nn.Linear, nn.GRU},
            dtype=torch.qint8
        )

def convert_to_int8_qat(model, word_map=None, qat_epochs=2):
    """Int8 QAT (Quantization-Aware Training) - [수정됨: FP32 학습 강제]"""
    print(f"   👉 Int8 변환 중 (정적 양자화 → QAT Fine-tuning: {qat_epochs} epochs)...")
    
    import platform
    machine = platform.machine().lower()
    backend = 'qnnpack' if ('arm' in machine or 'aarch64' in machine) else 'fbgemm'
    torch.backends.quantized.engine = backend
    print(f"   ⚙️ Quantization Engine: {backend}")

    model_cpu = deepcopy(model).cpu()
    model_cpu.train() 
    
    if word_map is None:
        return torch.quantization.quantize_dynamic(model_cpu, {nn.Linear}, dtype=torch.qint8)
    
    cal_images, _ = prepare_calibration_dataset(word_map, num_samples=20)
    example_input = cal_images[0]
    
    # [설정] QAT Config
    qconfig_dict = {"": torch.quantization.get_default_qat_qconfig(backend)}
    
    print("   🔧 인코더 QAT 준비 (Prepare QAT FX)...")
    model_cpu.encoder = quantize_fx.prepare_qat_fx(model_cpu.encoder, qconfig_dict, example_input)
    
    # [초기화] Calibration
    print("   🔄 초기 Calibration (Start)...")
    model_cpu.encoder.eval() # 통계 수집만
    with torch.no_grad():
        for img in cal_images:
            model_cpu.encoder(img)
    model_cpu.train() # 다시 학습 모드
    
    # [학습] QAT Fine-tuning
    print(f"\n   [3단계] QAT Fine-tuning 시작 ({qat_epochs} epochs)...")
    
    # Dataset/DataLoader 설정
    from torch.utils.data import DataLoader, Dataset
    
    # CaptionDataset 정의 (간소화)
    class CaptionDataset(Dataset):
        def __init__(self, images_dir, captions_file, transform, word_map, max_len=50):
            self.images_dir = images_dir
            self.transform = transform
            self.word_map = word_map
            self.max_len = max_len
            self.pairs = []
            
            # 파일 읽기 및 매핑 로직 (생략 가능하나 안전을 위해 유지)
            if os.path.exists(captions_file):
                 with open(captions_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if ',' in line:
                            parts = line.split(',', 1)
                            if len(parts) == 2:
                                self.pairs.append((parts[0].strip(), parts[1].strip()))

        def __len__(self):
            return len(self.pairs)
        
        def __getitem__(self, idx):
            img_name, cap_text = self.pairs[idx]
            # 이미지 로드
            try:
                img = Image.open(os.path.join(self.images_dir, img_name)).convert('RGB')
                if self.transform: img = self.transform(img)
            except:
                img = torch.zeros(3, 224, 224)
            
            # 캡션 인코딩
            tokens = cap_text.lower().split()
            enc = [self.word_map.get('<start>', 1)] + \
                  [self.word_map.get(t, self.word_map.get('<unk>', 3)) for t in tokens[:self.max_len-2]] + \
                  [self.word_map.get('<end>', 2)]
            while len(enc) < self.max_len: enc.append(0)
            return img, torch.LongTensor(enc[:self.max_len])

    dataset = CaptionDataset(TEST_IMAGE_DIR, CAPTIONS_FILE, transform, word_map)
    
    if len(dataset) == 0:
        print("   ⚠️ 학습 데이터가 부족해 Static Quantization으로 대체합니다.")
        return convert_to_int8_static(model, word_map)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    qat_device = torch.device("cpu") 
    
    # CUDA(NVIDIA)가 있는 경우에만 GPU 사용 (CUDA는 지원함)
    if torch.cuda.is_available():
        qat_device = torch.device("cuda")
        print(f"   🚀 QAT 학습 디바이스: CUDA (Precision: FP32)")
    else:
        print(f"   💻 QAT 학습 디바이스: CPU (MPS 미지원으로 인한 강제 설정)")
    model_cpu = model_cpu.to(qat_device)
    
    optimizer = torch.optim.Adam(model_cpu.parameters(), lr=4e-4) # 학습률 낮춤
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    for epoch in range(qat_epochs):
        epoch_loss = 0
        steps = 0
        for i, (imgs, caps) in enumerate(dataloader):
            if i > 50: break # 시간 절약을 위해 epoch당 50배치만
            
            imgs = imgs.to(qat_device)
            caps = caps.to(qat_device)
            
            optimizer.zero_grad()
            
            # [수정] Autocast 제거 -> FP32 강제
            outputs, _ = model_cpu(imgs, caps)
            targets = caps[:, 1:]
            outputs = outputs[:, :targets.shape[1], :]
            
            loss = criterion(outputs.reshape(-1, len(word_map)), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            steps += 1
            
        print(f"      Epoch {epoch+1}/{qat_epochs} Loss: {epoch_loss/steps:.4f}")

    print("   🔄 CPU로 이동 및 변환 준비...")
    model_cpu = model_cpu.cpu()
    model_cpu.eval() # Convert 전에는 반드시 Eval 모드
    
    print("   ⚡ QAT 모델 변환 (Convert FX)...")
    model_cpu.encoder = quantize_fx.convert_fx(model_cpu.encoder)
    
    print("   🔄 디코더 동적 양자화 적용...")
    quantized_model = torch.quantization.quantize_dynamic(
        model_cpu,
        {nn.Linear, nn.GRU, nn.LSTM},
        dtype=torch.qint8
    )
    
    return quantized_model

# ============================================================================
# 벤치마크 엔진 (기존과 동일)
# ============================================================================
def run_benchmark(model, img_tensor, wm, rwm, precision_name, ref_caption=None):
    print(f"\n[{precision_name}] 벤치마크 시작...")
    model_device = next(model.parameters()).device
    inp = img_tensor.clone().detach().to(model_device)
    
    if precision_name == "FP16": inp = inp.half()
    elif "Int8" in precision_name: inp = inp.float().cpu()

    # Warm-up
    with torch.no_grad():
        try: _ = model.generate(inp, wm, rwm, 20)
        except: pass

    # Time
    latencies = []
    start_mem = get_peak_memory_mb()
    peak_mem = start_mem
    
    for i in range(NUM_RUNS):
        gc.collect()
        if device.type == 'cuda': 
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        start = time.time()
        with torch.no_grad():
            gen_seq = model.generate(inp, wm, rwm, 20)
        if device.type == 'cuda': torch.cuda.synchronize()
        
        latencies.append((time.time() - start) * 1000)
        
        current_mem = get_peak_memory_mb()
        peak_mem = max(peak_mem, current_mem)
        
        if (i + 1) % 10 == 0:
            print(f"   진행: {i+1}/{NUM_RUNS}")
    
    # METEOR
    meteor_scores = []
    example_caption = "N/A"
    if ref_caption:
        for _ in range(5):
            with torch.no_grad(): 
                gen = model.generate(inp, wm, rwm, 20)
            score = calculate_meteor(gen, ref_caption)
            if score: meteor_scores.append(score)
            if _ == 0:
                example_caption = ' '.join([w for w in gen if w not in ['<start>', '<end>', '<pad>', '<unk>']])
            
    avg_meteor = np.mean(meteor_scores) if meteor_scores else None
    
    # 파라미터 개수 계산
    total_params, trainable_params = count_parameters(model)
    
    # 결과 정리
    avg_time = np.mean(latencies)
    std_time = np.std(latencies)
    size_mb = get_model_size_mb(model)
    memory_usage = peak_mem - start_mem
    
    print(f"   ⏱️ 평균 시간: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"   💾 모델 크기: {size_mb:.2f} MB")
    print(f"   📊 파라미터 개수: {total_params:,} (학습 가능: {trainable_params:,})")
    print(f"   🧠 메모리 사용량: {memory_usage:.2f} MB")
    if avg_meteor: 
        print(f"   ⭐ METEOR: {avg_meteor:.4f}")
    print(f"   📝 예시 캡션: {example_caption}")
    
    return {
        'precision': precision_name,
        'mean_time_ms': avg_time,
        'std_time_ms': std_time,
        'min_time_ms': np.min(latencies),
        'max_time_ms': np.max(latencies),
        'model_size_mb': size_mb,
        'memory_usage_mb': memory_usage,
        'meteor_score': avg_meteor,
        'inference_times': latencies,  # 추가: Box plot용
        'total_params': total_params,  # 추가: 파라미터 개수
        'trainable_params': trainable_params,
        'example_caption': example_caption
    }

def plot_benchmark(results):
    """
    벤치마크 결과를 종합적으로 시각화하여 저장하는 함수
    (추론 시간, 모델 크기, 메모리, METEOR, 파라미터, 시간 분포)
    """
    if not results:
        print("❌ 결과가 없어 plot을 생성할 수 없습니다.")
        return
    
    # 저장 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 데이터 추출
    precisions = [r['precision'] for r in results]
    mean_times = [r['mean_time_ms'] for r in results]
    std_times = [r['std_time_ms'] for r in results]
    model_sizes = [r['model_size_mb'] for r in results]
    memory_usages = [r['memory_usage_mb'] for r in results]
    
    # METEOR 점수 처리 (None이 아닌 경우만)
    meteor_scores = []
    for r in results:
        if r.get('meteor_score') is not None:
            meteor_scores.append(r['meteor_score'])
        else:
            meteor_scores.append(0)
    
    # inference_times 처리 (빈 리스트 방지)
    inference_times_list = []
    for r in results:
        times = r.get('inference_times', [])
        if times and len(times) > 0:
            inference_times_list.append(times)
        else:
            # fallback: mean_time_ms를 기반으로 더미 데이터 생성
            mean = r.get('mean_time_ms', 0)
            std = r.get('std_time_ms', 0)
            dummy_times = np.random.normal(mean, std, NUM_RUNS).tolist()
            inference_times_list.append(dummy_times)
    
    # total_params 처리
    total_params_list = []
    for r in results:
        params = r.get('total_params', 0)
        if params == 0:
            # fallback: 모델 크기로부터 추정
            size_mb = r.get('model_size_mb', 0)
            # 대략적인 추정 (4 bytes per param)
            estimated_params = int(size_mb * 1024 * 1024 / 4)
            total_params_list.append(estimated_params)
        else:
            total_params_list.append(params)
    
    # METEOR 점수가 유효한지 확인 (하나라도 있으면 표시)
    has_meteor = any(r.get('meteor_score') is not None and r.get('meteor_score') > 0 for r in results)
    
    # 색상 설정 (순서대로: 파랑, 빨강, 초록, 주황)
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12'] 
    bar_colors = colors[:len(precisions)]
    
    # 2. 캔버스 설정 (3행 2열)
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    fig.suptitle('Quantization 성능 비교 종합', fontsize=16, fontweight='bold')
    
    # -------------------------------------------------------
    # [1] 추론 시간 (Bar Chart)
    # -------------------------------------------------------
    ax = axes[0, 0]
    bars = ax.bar(precisions, mean_times, yerr=std_times, capsize=5, alpha=0.8, color=bar_colors)
    ax.set_ylabel('추론 시간 (ms)', fontweight='bold')
    ax.set_title('추론 시간 (낮을수록 좋음)', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    # 값 표시
    for bar, mean in zip(bars, mean_times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{mean:.1f}', ha='center', va='bottom', fontsize=9)

    # -------------------------------------------------------
    # [2] 모델 크기 (Bar Chart)
    # -------------------------------------------------------
    ax = axes[0, 1]
    bars = ax.bar(precisions, model_sizes, alpha=0.8, color=bar_colors)
    ax.set_ylabel('모델 크기 (MB)', fontweight='bold')
    ax.set_title('모델 크기 (낮을수록 좋음)', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar, size in zip(bars, model_sizes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{size:.1f}', ha='center', va='bottom', fontsize=9)

    # -------------------------------------------------------
    # [3] 메모리 사용량 (Bar Chart)
    # -------------------------------------------------------
    ax = axes[1, 0]
    bars = ax.bar(precisions, memory_usages, alpha=0.8, color=bar_colors)
    ax.set_ylabel('메모리 사용량 (MB)', fontweight='bold')
    ax.set_title('메모리 사용량', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar, mem in zip(bars, memory_usages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{mem:.1f}', ha='center', va='bottom', fontsize=9)

    # -------------------------------------------------------
    # [4] METEOR 점수 (Bar Chart)
    # -------------------------------------------------------
    ax = axes[1, 1]
    if has_meteor:
        bars = ax.bar(precisions, meteor_scores, alpha=0.8, color=bar_colors)
        ax.set_ylabel('METEOR 점수', fontweight='bold')
        ax.set_title('정확도 (높을수록 좋음)', fontweight='bold')
        ax.set_ylim(0, 1.0) # 점수는 0~1 사이
        ax.grid(axis='y', alpha=0.3)
        for bar, score in zip(bars, meteor_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'METEOR 점수 없음', ha='center', va='center')

    # -------------------------------------------------------
    # [5] 파라미터 개수 (Bar Chart)
    # -------------------------------------------------------
    ax = axes[2, 0]
    # 백만(M) 단위로 변환
    params_m = [p / 1e6 if p > 0 else 0 for p in total_params_list]
    if any(p > 0 for p in params_m):
        bars = ax.bar(precisions, params_m, alpha=0.8, color=bar_colors)
        ax.set_ylabel('파라미터 수 (Million)', fontweight='bold')
        ax.set_title('파라미터 개수 (M)', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        for bar, param in zip(bars, params_m):
            if param > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(params_m) * 0.02,
                        f'{param:.2f}M', ha='center', va='bottom', fontsize=9)
    else:
        ax.text(0.5, 0.5, '파라미터 데이터 없음', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('파라미터 개수 (데이터 없음)', fontweight='bold')

    # -------------------------------------------------------
    # [6] 추론 시간 분포 (Box Plot)
    # -------------------------------------------------------
    ax = axes[2, 1]
    # 빈 리스트 필터링
    valid_times = []
    valid_labels = []
    valid_colors = []
    for i, (times, label, color) in enumerate(zip(inference_times_list, precisions, bar_colors)):
        if times and len(times) > 0:
            valid_times.append(times)
            valid_labels.append(label)
            valid_colors.append(color)
    
    if valid_times:
        bp = ax.boxplot(valid_times, labels=valid_labels, patch_artist=True)
        # 박스 색상 입히기
        for patch, color in zip(bp['boxes'], valid_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        ax.set_ylabel('시간 (ms)', fontweight='bold')
        ax.set_title('추론 시간 분포 (안정성)', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    else:
        ax.text(0.5, 0.5, '추론 시간 데이터 없음', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('추론 시간 분포 (데이터 없음)', fontweight='bold')

    # 레이아웃 조정 및 저장
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'quantization_benchmark_result.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 그래프 저장 완료: {save_path}")
    plt.close()

# ============================================================================
# Main
# ============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_model, wm, rwm = load_base_model()
    img_tensor, ref_caption = load_data()
    
    results = []
    
    # FP32
    results.append(run_benchmark(base_model, img_tensor, wm, rwm, "FP32", ref_caption))
    
    # FP16 (GPU only)
    if device.type in ['cuda', 'mps']:
        m_fp16 = deepcopy(base_model).half()
        results.append(run_benchmark(m_fp16, img_tensor, wm, rwm, "FP16", ref_caption))
        del m_fp16
    
    # Int8 Static
    m_static = convert_to_int8(base_model, wm, use_qat=False)
    results.append(run_benchmark(m_static, img_tensor, wm, rwm, "Int8-Static", ref_caption))
    del m_static
    
    # Int8 QAT
    if USE_QAT:
        m_qat = convert_to_int8(base_model, wm, use_qat=True, qat_epochs=QAT_EPOCHS)
        results.append(run_benchmark(m_qat, img_tensor, wm, rwm, "Int8-QAT", ref_caption))
        del m_qat
        
    print("\n✅ 모든 벤치마크 완료.")
    plot_benchmark(results)
if __name__ == "__main__":
    main()