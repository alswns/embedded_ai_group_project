"""
QAT Fine-tuning 전용 스크립트
정적 양자화 후 QAT fine-tuning을 적용하고 결과를 비교합니다.
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

warnings.filterwarnings('ignore')

# -------------------------------------------------------------------------
# 모델 import
# -------------------------------------------------------------------------
try:
    from src.muti_modal_model.model import MobileNetCaptioningModel
except ImportError:
    print("⚠️ 모델 클래스를 import할 수 없습니다. 경로를 확인해주세요.")
    class MobileNetCaptioningModel(nn.Module):
        def __init__(self, vocab_size, embed_dim):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, embed_dim)
            self.gru = nn.GRU(embed_dim, 512)
            self.fc = nn.Linear(512, vocab_size)
        def generate(self, img, wm, rwm, max_len):
            return ["<start>", "a", "test", "caption", "<end>"]

# NLTK 및 METEOR 설정
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

# ============================================================================
# 설정
# ============================================================================
matplotlib.use('Agg')  # GUI 없이 백그라운드에서 실행

# 한글 폰트 설정
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

MODEL_PATH = "models/lightweight_captioning_model.pth"
TEST_IMAGE_DIR = "assets/images"
CAPTIONS_FILE = "assets/captions.txt"
OUTPUT_DIR = "qat_results"
NUM_RUNS = 50

# QAT 설정
QAT_EPOCHS = 3  # QAT 학습 epoch 수 (더 많은 학습으로 더 나은 결과)

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
    """모델 파라미터 개수 계산"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def get_model_size_mb(model):
    """모델 파라미터 + 버퍼 크기 계산 (MB)"""
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

def calculate_meteor(generated_caption, reference_caption):
    """METEOR 점수 계산"""
    if not METEOR_AVAILABLE:
        return None
    try:
        # 특수 토큰 제거
        gen_words = [w for w in generated_caption if w not in ['<start>', '<end>', '<pad>', '<unk>']]
        ref_words = word_tokenize(reference_caption.lower())
        gen_words_str = ' '.join(gen_words)
        if not gen_words_str:
            return None
        gen_tokens = word_tokenize(gen_words_str.lower())
        score = meteor_score([ref_words], gen_tokens)
        return score
    except Exception as e:
        return None

# ============================================================================
# 데이터 로드
# ============================================================================
def load_base_model():
    """학습된 모델 로드"""
    print("📂 모델 로드 중...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
    
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    # 체크포인트에서 정보 추출
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            vocab_size = checkpoint.get('vocab_size', 1000)
            word_map = checkpoint.get('word_map', {})
            rev_word_map = checkpoint.get('rev_word_map', {})
        else:
            model_state = checkpoint
            vocab_size = 1000
            word_map = {}
            rev_word_map = {}
    else:
        model_state = checkpoint
        vocab_size = 1000
        word_map = {}
        rev_word_map = {}
    
    # 모델 생성
    embed_dim = 300  # GloVe 사용 시
    model = MobileNetCaptioningModel(vocab_size=vocab_size, embed_dim=embed_dim)
    model.load_state_dict(model_state)
    model.eval()
    model.to(device)
    
    print(f"✅ 모델 로드 완료 (Vocab Size: {vocab_size})")
    return model, word_map, rev_word_map

def load_data():
    """테스트 이미지와 참조 캡션 로드"""
    img_tensor = None
    filename = None
    ref_caption = None
    
    if os.path.exists(TEST_IMAGE_DIR):
        files = [f for f in os.listdir(TEST_IMAGE_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if files:
            import random
            filename = random.choice(files)
            img_path = os.path.join(TEST_IMAGE_DIR, filename)
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            print(f"📸 테스트 이미지: {filename}")
    
    if img_tensor is None:
        print("⚠️ 이미지를 찾을 수 없어 더미 데이터를 사용합니다.")
        img_tensor = torch.randn(1, 3, 224, 224).to(device)
        filename = "dummy"
        ref_caption = "a test image"
    else:
        # 참조 캡션 로드
        if os.path.exists(CAPTIONS_FILE) and filename != "dummy":
            with open(CAPTIONS_FILE, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    if ',' in line:
                        parts = line.split(',', 1)
                        if len(parts) == 2 and parts[0].strip() == filename:
                            ref_caption = parts[1].strip()
                            print(f"📝 참조 캡션: {ref_caption}")
                            break
    
    return img_tensor, ref_caption

# ============================================================================
# Calibration 데이터셋 준비
# ============================================================================
def prepare_calibration_dataset(word_map, num_samples=100):
    """정적 양자화를 위한 Calibration 데이터셋 준비"""
    calibration_images = []
    calibration_captions = []
    
    if not os.path.exists(TEST_IMAGE_DIR):
        print(f"   ⚠️ 이미지 디렉토리가 없어 더미 데이터를 사용합니다.")
        for _ in range(num_samples):
            dummy_img = torch.randn(1, 3, 224, 224)
            calibration_images.append(dummy_img)
            dummy_cap = torch.LongTensor([
                word_map.get('<start>', 1),
                word_map.get('<pad>', 0),
                word_map.get('<end>', 2)
            ])
            calibration_captions.append(dummy_cap)
        return calibration_images, calibration_captions
    
    image_files = [f for f in os.listdir(TEST_IMAGE_DIR) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"   ⚠️ 이미지가 없어 더미 데이터를 사용합니다.")
        for _ in range(num_samples):
            dummy_img = torch.randn(1, 3, 224, 224)
            calibration_images.append(dummy_img)
            dummy_cap = torch.LongTensor([
                word_map.get('<start>', 1),
                word_map.get('<pad>', 0),
                word_map.get('<end>', 2)
            ])
            calibration_captions.append(dummy_cap)
        return calibration_images, calibration_captions
    
    import random
    selected_files = random.sample(image_files, min(num_samples, len(image_files)))
    
    print(f"   📊 Calibration 데이터셋 준비 중: {len(selected_files)}개 이미지")
    
    for filename in selected_files:
        try:
            img_path = os.path.join(TEST_IMAGE_DIR, filename)
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)
            calibration_images.append(img_tensor)
            
            dummy_cap = torch.LongTensor([
                word_map.get('<start>', 1),
                word_map.get('<pad>', 0),
                word_map.get('<end>', 2)
            ])
            calibration_captions.append(dummy_cap)
        except Exception as e:
            print(f"   ⚠️ 이미지 로드 실패 ({filename}): {e}")
            continue
    
    while len(calibration_images) < num_samples:
        dummy_img = torch.randn(1, 3, 224, 224)
        calibration_images.append(dummy_img)
        dummy_cap = torch.LongTensor([
            word_map.get('<start>', 1),
            word_map.get('<pad>', 0),
            word_map.get('<end>', 2)
        ])
        calibration_captions.append(dummy_cap)
    
    return calibration_images[:num_samples], calibration_captions[:num_samples]

# ============================================================================
# Quantization 함수
# ============================================================================
def convert_to_int8_static(model, word_map=None):
    """Int8 Static Quantization"""
    print("   👉 Int8 정적 양자화 적용 중...")
    
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
        
        print("   🔧 인코더 준비 (Prepare FX)...")
        model_cpu.encoder = quantize_fx.prepare_fx(model_cpu.encoder, qconfig_dict, example_input)

        print("   🔄 Calibration 진행 중...")
        with torch.no_grad():
            for i, img in enumerate(cal_images):
                model_cpu.encoder(img)

        print("   ⚡ 인코더 변환 (Convert FX)...")
        model_cpu.encoder = quantize_fx.convert_fx(model_cpu.encoder)

        print("   🔄 디코더 동적 양자화 적용...")
        quantized_model = torch.quantization.quantize_dynamic(
            model_cpu,
            {nn.Linear, nn.GRU, nn.LSTM},
            dtype=torch.qint8
        )
        
        print("   ✅ 정적 양자화 완료!")
        return quantized_model

    except Exception as e:
        print(f"   ⚠️ 정적 양자화 실패: {e}")
        return torch.quantization.quantize_dynamic(
            deepcopy(model).cpu(),
            {nn.Linear, nn.GRU},
            dtype=torch.qint8
        )

def convert_to_int8_qat(model, word_map=None, qat_epochs=3):
    """Int8 QAT (Quantization-Aware Training)"""
    print(f"   👉 Int8 QAT 적용 중 ({qat_epochs} epochs)...")
    
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
    model_cpu.train()

    if word_map is None:
        print("   ⚠️ word_map이 없어 Dynamic Quantization으로 fallback")
        return torch.quantization.quantize_dynamic(model_cpu, {nn.Linear}, dtype=torch.qint8)

    print("   📊 Calibration 데이터 준비 중...")
    cal_images, _ = prepare_calibration_dataset(word_map, num_samples=20)
    example_input = cal_images[0]
    
    qconfig_dict = {"": torch.quantization.get_default_qat_qconfig(backend)}
    
    print("   🔧 인코더 QAT 준비 (Prepare QAT FX)...")
    model_cpu.encoder = quantize_fx.prepare_qat_fx(
        model_cpu.encoder,
        qconfig_dict,
        example_input
    )
    
    print("   🔄 Calibration 진행 중 (초기 양자화 파라미터 설정)...")
    model_cpu.encoder.eval()
    with torch.no_grad():
        for img in cal_images:
            model_cpu.encoder(img)
    
    print(f"\n   [QAT Fine-tuning 시작]")
    model_cpu.train()
    
    # 학습 데이터셋 준비
    try:
        from torch.utils.data import DataLoader, Dataset
        
        MAX_CAPTION_LEN = 50
        
        def encode_caption(caption, word_map, max_len=MAX_CAPTION_LEN):
            tokens = caption.lower().split()
            encoded = [word_map.get('<start>', 1)]
            for token in tokens[:max_len-2]:
                encoded.append(word_map.get(token, word_map.get('<unk>', 3)))
            encoded.append(word_map.get('<end>', 2))
            while len(encoded) < max_len:
                encoded.append(word_map.get('<pad>', 0))
            return torch.LongTensor(encoded[:max_len])
        
        class CaptionDataset(Dataset):
            def __init__(self, images_dir, captions_file, transform=None, word_map=None, max_len=MAX_CAPTION_LEN):
                self.images_dir = images_dir
                self.transform = transform
                self.word_map = word_map
                self.max_len = max_len
                
                available_images = set([f for f in os.listdir(images_dir) 
                                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
                
                image_to_captions = defaultdict(list)
                if os.path.exists(captions_file):
                    with open(captions_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        first_line = lines[0].strip() if lines else ""
                        start_idx = 1 if first_line.lower().startswith('image') or first_line.lower().startswith('filename') else 0
                        
                        for line in lines[start_idx:]:
                            line = line.strip()
                            if not line:
                                continue
                            if ',' in line:
                                parts = line.split(',', 1)
                                if len(parts) == 2:
                                    img_name = parts[0].strip()
                                    caption = parts[1].strip()
                                    if img_name and caption and img_name in available_images:
                                        image_to_captions[img_name].append(caption)
                
                self.image_caption_pairs = []
                for img_name, captions in image_to_captions.items():
                    if captions:
                        for caption in captions:
                            self.image_caption_pairs.append((img_name, caption))
            
            def __getitem__(self, idx):
                img_name, caption_text = self.image_caption_pairs[idx]
                img_path = os.path.join(self.images_dir, img_name)
                try:
                    image = Image.open(img_path).convert('RGB')
                    if self.transform:
                        image = self.transform(image)
                except Exception:
                    image = torch.zeros(3, 224, 224)
                
                if self.word_map:
                    caption = encode_caption(caption_text, self.word_map, self.max_len)
                else:
                    caption = torch.zeros(self.max_len, dtype=torch.long)
                return image, caption
            
            def __len__(self):
                return len(self.image_caption_pairs)
        
        dataset = CaptionDataset(
            images_dir=TEST_IMAGE_DIR,
            captions_file=CAPTIONS_FILE,
            transform=transform,
            word_map=word_map,
            max_len=MAX_CAPTION_LEN
        )
        
        if len(dataset) == 0:
            print("   ⚠️ 학습 데이터가 없어 Static Quantization으로 fallback")
            return convert_to_int8_static(model, word_map)
        
        dataloader = DataLoader(
            dataset, 
            batch_size=4, 
            shuffle=True, 
            num_workers=0,
            pin_memory=False
        )
        
        print(f"   📚 학습 데이터: {len(dataset)}개 샘플")
        
        # Mixed Precision 설정
        use_mixed_precision = False
        scaler = None
        qat_device = torch.device("cpu")
        
        if torch.cuda.is_available():
            qat_device = torch.device("cuda")
            model_cpu = model_cpu.to(qat_device)
            use_mixed_precision = True
            scaler = torch.cuda.amp.GradScaler()
            print("   🚀 GPU 사용 - FP16 Mixed Precision 활성화")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            qat_device = torch.device("mps")
            model_cpu = model_cpu.to(qat_device)
            use_mixed_precision = True
            print("   🚀 MPS 사용 - FP16 Mixed Precision 활성화")
        else:
            print("   💻 CPU 사용 - FP32 학습")
        
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = torch.optim.Adam(model_cpu.parameters(), lr=1e-4)
        vocab_size = len(word_map)
        
        for epoch in range(qat_epochs):
            epoch_loss = 0
            num_batches = 0
            
            for batch_idx, (imgs, caps) in enumerate(dataloader):
                if batch_idx >= 30:  # 더 많은 배치로 학습
                    break
                
                imgs = imgs.to(qat_device)
                caps = caps.to(qat_device)
                
                optimizer.zero_grad()
                
                try:
                    if use_mixed_precision:
                        if qat_device.type == "cuda" and scaler is not None:
                            with torch.cuda.amp.autocast():
                                outputs, alphas = model_cpu(imgs, caps)
                                targets = caps[:, 1:]
                                outputs = outputs[:, :targets.shape[1], :]
                                loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
                            
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        elif qat_device.type == "mps":
                            with torch.amp.autocast(device_type="mps", dtype=torch.float16):
                                outputs, alphas = model_cpu(imgs, caps)
                                targets = caps[:, 1:]
                                outputs = outputs[:, :targets.shape[1], :]
                                loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
                            
                            loss.backward()
                            optimizer.step()
                    else:
                        outputs, alphas = model_cpu(imgs, caps)
                        targets = caps[:, 1:]
                        outputs = outputs[:, :targets.shape[1], :]
                        loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
                        loss.backward()
                        optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                except Exception as e:
                    print(f"   ⚠️ 배치 {batch_idx} 학습 실패: {e}")
                    continue
            
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                print(f"      Epoch {epoch+1}/{qat_epochs}, Loss: {avg_loss:.4f}")
        
        print("   🔄 CPU로 이동 중 (Quantization 준비)...")
        model_cpu = model_cpu.cpu()
        
        print("   ⚡ QAT 모델 변환 (Convert FX)...")
        model_cpu.eval()
        model_cpu.encoder = quantize_fx.convert_fx(model_cpu.encoder)
        
        print("   🔄 디코더 동적 양자화 적용...")
        quantized_model = torch.quantization.quantize_dynamic(
            model_cpu,
            {nn.Linear, nn.GRU, nn.LSTM},
            dtype=torch.qint8
        )
        quantized_model.eval()
        
        print("   ✅ QAT 완료!")
        return quantized_model
        
    except Exception as e:
        print(f"   ⚠️ QAT 실패: {e}")
        import traceback
        traceback.print_exc()
        return convert_to_int8_static(model, word_map)

# ============================================================================
# 벤치마크 엔진
# ============================================================================
def run_benchmark(model, img_tensor, wm, rwm, precision_name, ref_caption=None):
    """벤치마크 실행"""
    print(f"\n[{precision_name}] 벤치마크 시작...")
    
    model_device = next(model.parameters()).device
    inp = img_tensor.clone().detach().to(model_device)
    
    if "Int8" in precision_name:
        inp = inp.float().cpu()
    
    # Warm-up
    with torch.no_grad():
        try:
            _ = model.generate(inp, wm, rwm, 20)
        except Exception as e:
            print(f"⚠️ Warm-up 실패: {e}")
            return None
    
    # 속도 측정
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
            
        if device.type == 'cuda': 
            torch.cuda.synchronize()
        
        latencies.append((time.time() - start) * 1000)
        
        current_mem = get_peak_memory_mb()
        peak_mem = max(peak_mem, current_mem)
        
        if (i + 1) % 10 == 0:
            print(f"   진행: {i+1}/{NUM_RUNS}")
    
    # METEOR 점수 계산
    meteor_scores = []
    example_caption = "N/A"
    
    if ref_caption:
        for _ in range(5):
            with torch.no_grad():
                gen_seq = model.generate(inp, wm, rwm, 20)
            meteor = calculate_meteor(gen_seq, ref_caption)
            if meteor is not None:
                meteor_scores.append(meteor)
            if _ == 0:
                example_caption = ' '.join([w for w in gen_seq if w not in ['<start>', '<end>', '<pad>', '<unk>']])
    
    avg_meteor = np.mean(meteor_scores) if meteor_scores else None
    
    # 결과 정리
    avg_time = np.mean(latencies)
    std_time = np.std(latencies)
    size_mb = get_model_size_mb(model)
    memory_usage = peak_mem - start_mem
    total_params, trainable_params = count_parameters(model)
    
    print(f"   ⏱️ 평균 시간: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"   💾 모델 크기: {size_mb:.2f} MB")
    print(f"   📊 파라미터 개수: {total_params:,} (학습 가능: {trainable_params:,})")
    print(f"   🧠 메모리 사용량: {memory_usage:.2f} MB")
    if avg_meteor is not None:
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
        'inference_times': latencies,
        'example_caption': example_caption,
        'total_params': total_params,
        'trainable_params': trainable_params
    }

# ============================================================================
# 시각화
# ============================================================================
def plot_qat_comparison(result_static, result_qat):
    """QAT Fine-tuning 전후 비교 그래프"""
    if not result_static or not result_qat:
        return
    
    metrics = []
    static_values = []
    qat_values = []
    improvements = []
    
    # 추론 시간
    static_time = result_static['mean_time_ms']
    qat_time = result_qat['mean_time_ms']
    time_improvement = ((static_time - qat_time) / static_time) * 100
    metrics.append('추론 시간\n(ms)')
    static_values.append(static_time)
    qat_values.append(qat_time)
    improvements.append(time_improvement)
    
    # METEOR 점수
    if result_static.get('meteor_score') and result_qat.get('meteor_score'):
        static_meteor = result_static['meteor_score']
        qat_meteor = result_qat['meteor_score']
        meteor_improvement = ((qat_meteor - static_meteor) / static_meteor) * 100
        metrics.append('METEOR\n점수')
        static_values.append(static_meteor * 100)
        qat_values.append(qat_meteor * 100)
        improvements.append(meteor_improvement)
    
    # 메모리 사용량
    static_mem = result_static['memory_usage_mb']
    qat_mem = result_qat['memory_usage_mb']
    mem_improvement = ((static_mem - qat_mem) / static_mem) * 100 if static_mem > 0 else 0
    metrics.append('메모리\n(MB)')
    static_values.append(static_mem)
    qat_values.append(qat_mem)
    improvements.append(mem_improvement)
    
    # 그래프 생성
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('QAT Fine-tuning 전후 비교', fontsize=16, fontweight='bold')
    
    x = np.arange(len(metrics))
    width = 0.35
    
    # 1. 값 비교
    bars1 = ax1.bar(x - width/2, static_values, width, label='Static (Before)', alpha=0.8, color='#e74c3c')
    bars2 = ax1.bar(x + width/2, qat_values, width, label='QAT (After)', alpha=0.8, color='#2ecc71')
    
    ax1.set_ylabel('값', fontweight='bold')
    ax1.set_title('값 비교', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=9)
    
    # 2. 개선율
    colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]
    bars3 = ax2.bar(metrics, improvements, alpha=0.8, color=colors)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('개선율 (%)', fontweight='bold')
    ax2.set_title('개선율', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, imp in zip(bars3, improvements):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{imp:+.2f}%',
                ha='center', va='bottom' if imp > 0 else 'top', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'qat_fine_tuning_comparison.png'), 
                dpi=300, bbox_inches='tight')
    print(f"✅ QAT 비교 Plot 저장: {os.path.join(OUTPUT_DIR, 'qat_fine_tuning_comparison.png')}")
    plt.close()

# ============================================================================
# Main
# ============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*70)
    print("=== QAT Fine-tuning 벤치마크 ===")
    print("="*70)
    
    # 1. 모델 및 데이터 로드
    base_model, wm, rwm = load_base_model()
    img_tensor, ref_caption = load_data()
    
    # 2. Int8 Static Quantization (Before)
    print("\n" + "="*70)
    print("=== [1단계] Int8 Static Quantization (Before Fine-tuning) ===")
    print("="*70)
    model_int8_static = convert_to_int8_static(base_model, wm)
    result_int8_static = run_benchmark(model_int8_static, img_tensor, wm, rwm, "Int8-Static (CPU)", ref_caption)
    del model_int8_static
    gc.collect()
    
    # 3. Int8 QAT (After Fine-tuning)
    print("\n" + "="*70)
    print("=== [2단계] Int8 QAT (After Fine-tuning) ===")
    print("="*70)
    model_int8_qat = convert_to_int8_qat(base_model, wm, qat_epochs=QAT_EPOCHS)
    result_int8_qat = run_benchmark(model_int8_qat, img_tensor, wm, rwm, "Int8-QAT (CPU)", ref_caption)
    del model_int8_qat
    gc.collect()
    
    # 4. 결과 비교 출력
    if result_int8_static and result_int8_qat:
        print("\n" + "="*70)
        print("=== 🎯 QAT Fine-tuning 전후 비교 결과 ===")
        print("="*70)
        print(f"{'Metric':<30} {'Static (Before)':<20} {'QAT (After)':<20} {'개선율':<15}")
        print("-"*85)
        
        # 추론 시간
        static_time = result_int8_static['mean_time_ms']
        qat_time = result_int8_qat['mean_time_ms']
        time_improvement = ((static_time - qat_time) / static_time) * 100
        time_emoji = "✅" if time_improvement > 0 else "❌"
        print(f"{'⏱️  추론 시간 (ms)':<30} {static_time:<20.2f} {qat_time:<20.2f} {time_emoji} {time_improvement:>8.2f}%")
        
        # METEOR 점수
        if result_int8_static.get('meteor_score') and result_int8_qat.get('meteor_score'):
            static_meteor = result_int8_static['meteor_score']
            qat_meteor = result_int8_qat['meteor_score']
            meteor_improvement = ((qat_meteor - static_meteor) / static_meteor) * 100
            meteor_emoji = "✅" if meteor_improvement > 0 else "❌"
            print(f"{'⭐ METEOR 점수':<30} {static_meteor:<20.4f} {qat_meteor:<20.4f} {meteor_emoji} {meteor_improvement:>8.2f}%")
        
        # 모델 크기
        static_size = result_int8_static['model_size_mb']
        qat_size = result_int8_qat['model_size_mb']
        print(f"{'💾 모델 크기 (MB)':<30} {static_size:<20.2f} {qat_size:<20.2f} {'-':>15}")
        
        # 메모리 사용량
        static_mem = result_int8_static['memory_usage_mb']
        qat_mem = result_int8_qat['memory_usage_mb']
        mem_improvement = ((static_mem - qat_mem) / static_mem) * 100 if static_mem > 0 else 0
        mem_emoji = "✅" if mem_improvement > 0 else "❌"
        print(f"{'🧠 메모리 사용량 (MB)':<30} {static_mem:<20.2f} {qat_mem:<20.2f} {mem_emoji} {mem_improvement:>8.2f}%")
        
        # 파라미터 개수
        static_params = result_int8_static.get('total_params', 0)
        qat_params = result_int8_qat.get('total_params', 0)
        static_params_m = static_params / 1e6
        qat_params_m = qat_params / 1e6
        print(f"{'📊 파라미터 개수 (M)':<30} {static_params_m:<20.2f} {qat_params_m:<20.2f} {'-':>15}")
        
        print("="*85)
        print("\n💡 해석:")
        if time_improvement > 0:
            print(f"   ✅ 추론 시간이 {time_improvement:.2f}% 개선되었습니다 (빠름)")
        else:
            print(f"   ⚠️ 추론 시간이 {abs(time_improvement):.2f}% 느려졌습니다")
        
        if result_int8_static.get('meteor_score') and result_int8_qat.get('meteor_score'):
            if meteor_improvement > 0:
                print(f"   ✅ METEOR 점수가 {meteor_improvement:.2f}% 개선되었습니다 (정확도 향상)")
            else:
                print(f"   ⚠️ METEOR 점수가 {abs(meteor_improvement):.2f}% 감소했습니다")
        
        if mem_improvement > 0:
            print(f"   ✅ 메모리 사용량이 {mem_improvement:.2f}% 감소했습니다 (효율적)")
        else:
            print(f"   ⚠️ 메모리 사용량이 {abs(mem_improvement):.2f}% 증가했습니다")
        
        print("="*85)
        
        # 그래프 생성
        print("\n📊 그래프 생성 중...")
        plot_qat_comparison(result_int8_static, result_int8_qat)
    
    print("\n" + "="*70)
    print("=== 벤치마크 완료 ===")
    print(f"결과 저장 위치: {OUTPUT_DIR}")
    print("="*70)

if __name__ == "__main__":
    main()

