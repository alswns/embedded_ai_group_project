"""
Pruning 벤치마크 스크립트
다양한 Pruning 기법을 적용하고 성능을 비교합니다.
"""
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
import os
import time
import psutil
import matplotlib.pyplot as plt
import matplotlib
from copy import deepcopy
import gc
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
OUTPUT_DIR = "pruning_results"
NUM_RUNS = 50

# Pruning 설정
PRUNING_RATES = [0.1, 0.3, 0.5, 0.7]  # 10%, 30%, 50%, 70% 프루닝
PRUNING_METHODS = ['magnitude', 'structured']  # 프루닝 방법

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

def count_nonzero_parameters(model):
    """0이 아닌 파라미터 개수 계산 (프루닝 후)"""
    nonzero_params = 0
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
        nonzero_params += param.nonzero().size(0) if param.numel() > 0 else 0
    return nonzero_params, total_params

def get_model_size_mb(model, sparse=False):
    """모델 파라미터 + 버퍼 크기 계산 (MB)
    
    Args:
        model: 모델
        sparse: True면 실제 0이 아닌 파라미터만 계산 (Pruning 후 실제 크기)
    """
    param_size = 0
    buffer_size = 0
    
    if sparse:
        # Sparse format: 실제 0이 아닌 값만 계산
        for param in model.parameters():
            if param.is_sparse:
                # Sparse tensor인 경우
                param_size += param._values().numel() * param.element_size()
                # 인덱스도 저장해야 하므로 추가
                param_size += param._indices().numel() * param._indices().element_size()
            else:
                # Dense tensor인 경우 0이 아닌 값만 계산
                nonzero = param.nonzero()
                if nonzero.numel() > 0:
                    # 0이 아닌 값의 개수
                    nonzero_count = (param != 0).sum().item()
                    param_size += nonzero_count * param.element_size()
                    # 인덱스 저장을 위한 오버헤드 (간단한 추정)
                    # 실제로는 더 효율적인 인코딩이 가능하지만, 여기서는 보수적으로 추정
                    param_size += nonzero_count * 4  # 인덱스 저장 오버헤드 (4 bytes per index)
    else:
        # Dense format: 모든 파라미터 계산
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    return (param_size + buffer_size) / 1024 / 1024

def convert_to_sparse_model(model):
    """Pruning된 모델을 실제로 sparse format으로 변환하여 크기 감소"""
    # 주의: 실제로 모델 구조를 변경하는 것은 복잡하므로
    # 여기서는 가중치를 sparse tensor로 변환하는 대신
    # 실제 0이 아닌 파라미터만 계산하는 방식 사용
    # 실제 배포 시에는 sparse format으로 저장/로드하는 것이 좋습니다
    return model

def save_sparse_model(model, path):
    """모델을 sparse format으로 저장 (실제 크기 감소)"""
    state_dict = {}
    for name, param in model.named_parameters():
        if param.numel() > 0:
            # 0이 아닌 값만 저장
            nonzero_mask = param != 0
            if nonzero_mask.any():
                # Sparse format으로 저장
                sparse_param = param[nonzero_mask]
                indices = nonzero_mask.nonzero(as_tuple=False)
                state_dict[name] = {
                    'values': sparse_param.cpu(),
                    'indices': indices.cpu(),
                    'shape': list(param.shape),
                    'dtype': str(param.dtype)
                }
            else:
                # 모든 값이 0인 경우
                state_dict[name] = {
                    'values': torch.tensor([], dtype=param.dtype),
                    'indices': torch.tensor([], dtype=torch.long),
                    'shape': list(param.shape),
                    'dtype': str(param.dtype)
                }
        else:
            state_dict[name] = param.cpu()
    
    # 버퍼도 저장
    for name, buffer in model.named_buffers():
        state_dict[name] = buffer.cpu()
    
    torch.save(state_dict, path)
    print(f"   💾 Sparse 모델 저장: {path}")

def get_sparse_model_size_mb(model):
    """Sparse format으로 저장했을 때의 실제 모델 크기 계산"""
    total_size = 0
    
    for name, param in model.named_parameters():
        if param.numel() > 0:
            # 0이 아닌 값의 개수
            nonzero_count = (param != 0).sum().item()
            total_params = param.numel()
            
            if nonzero_count > 0:
                # 값 저장 (0이 아닌 값만)
                total_size += nonzero_count * param.element_size()
                
                # 인덱스 저장 (COO format: Coordinate format)
                # 각 0이 아닌 값의 위치를 저장
                if len(param.shape) == 1:
                    # 1D: 인덱스만
                    indices_size = nonzero_count * 4  # 4 bytes per index
                elif len(param.shape) == 2:
                    # 2D: (row, col) 쌍
                    indices_size = nonzero_count * 2 * 4  # 2 indices per value
                else:
                    # 다차원: 모든 차원의 인덱스
                    indices_size = nonzero_count * len(param.shape) * 4
                
                total_size += indices_size
                
                # 메타데이터 (shape, dtype, nonzero_count 등)
                total_size += 64  # 메타데이터 오버헤드
            else:
                # 모든 값이 0인 경우 최소 메타데이터만
                total_size += 32
    
    # 버퍼 크기 (버퍼는 보통 작으므로 그대로 계산)
    for name, buffer in model.named_buffers():
        total_size += buffer.nelement() * buffer.element_size()
    
    return total_size / 1024 / 1024

def get_peak_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def calculate_meteor(generated_caption, reference_caption):
    """METEOR 점수 계산"""
    if not METEOR_AVAILABLE:
        return None
    try:
        gen_words = [w for w in generated_caption if w not in ['<start>', '<end>', '<pad>', '<unk>']]
        ref_words = word_tokenize(reference_caption.lower())
        gen_words_str = ' '.join(gen_words)
        if not gen_words_str:
            return None
        gen_tokens = word_tokenize(gen_words_str.lower())
        score = meteor_score([ref_words], gen_tokens)
        return score
    except Exception:
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
    
    embed_dim = 300
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
        ref_caption = "a test image"
    else:
        if os.path.exists(CAPTIONS_FILE) and filename != "dummy":
            with open(CAPTIONS_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    if filename in line:
                        if ',' in line:
                            parts = line.split(',', 1)
                            if len(parts) == 2:
                                ref_caption = parts[1].strip()
                                print(f"📝 참조 캡션: {ref_caption}")
                                break
    
    return img_tensor, ref_caption

# ============================================================================
# Pruning 함수
# ============================================================================
def apply_magnitude_pruning(model, pruning_rate):
    """Magnitude-based Pruning 적용 (가중치 크기 기반)"""
    pruned_model = deepcopy(model)
    pruned_model.eval()
    
    # 프루닝할 레이어 찾기 (Linear 레이어만)
    modules_to_prune = []
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Linear):
            modules_to_prune.append((module, 'weight'))
    
    # Magnitude-based pruning 적용
    for module, param_name in modules_to_prune:
        prune.l1_unstructured(module, name=param_name, amount=pruning_rate)
    
    # Pruning 영구 적용 (0으로 만들기)
    for module, param_name in modules_to_prune:
        prune.remove(module, param_name)
    
    # 실제로 0인 가중치를 제거하여 모델 크기 감소
    # 주의: 이는 메모리상에서만 효과가 있고, 실제 모델 구조는 변경되지 않음
    # 실제 배포 시에는 sparse format으로 저장/로드해야 함
    print(f"   ✂️ Pruning 완료: {pruning_rate*100:.0f}% 가중치 제거")
    
    return pruned_model

def apply_structured_pruning(model, pruning_rate):
    """Structured Pruning 적용 (채널/필터 단위)"""
    pruned_model = deepcopy(model)
    pruned_model.eval()
    
    # Structured pruning은 Linear 레이어에 적용
    modules_to_prune = []
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Linear):
            modules_to_prune.append((module, 'weight'))
    
    # Structured pruning 적용 (채널 단위)
    for module, param_name in modules_to_prune:
        prune.ln_structured(module, name=param_name, amount=pruning_rate, n=2, dim=0)
    
    # Pruning 영구 적용
    for module, param_name in modules_to_prune:
        prune.remove(module, param_name)
    
    return pruned_model

def apply_global_pruning(model, pruning_rate):
    """Global Pruning 적용 (전체 모델 기준)"""
    pruned_model = deepcopy(model)
    pruned_model.eval()
    
    # 프루닝할 파라미터 수집
    parameters_to_prune = []
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    
    # Global pruning 적용
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=pruning_rate,
    )
    
    # Pruning 영구 적용
    for module, param_name in parameters_to_prune:
        prune.remove(module, param_name)
    
    return pruned_model

# ============================================================================
# 벤치마크 엔진
# ============================================================================
def run_benchmark(model, img_tensor, wm, rwm, precision_name, ref_caption=None):
    """벤치마크 실행"""
    print(f"\n[{precision_name}] 벤치마크 시작...")
    
    model_device = next(model.parameters()).device
    inp = img_tensor.clone().detach().to(model_device)
    
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
    
    # Dense format 크기 (메모리상 크기)
    size_mb_dense = get_model_size_mb(model, sparse=False)
    # Sparse format 크기 (실제 저장 크기)
    size_mb_sparse = get_sparse_model_size_mb(model)
    
    memory_usage = peak_mem - start_mem
    total_params, trainable_params = count_parameters(model)
    nonzero_params, _ = count_nonzero_parameters(model)
    sparsity = 1.0 - (nonzero_params / total_params) if total_params > 0 else 0.0
    
    print(f"   ⏱️ 평균 시간: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"   💾 모델 크기 (Dense): {size_mb_dense:.2f} MB")
    print(f"   💾 모델 크기 (Sparse): {size_mb_sparse:.2f} MB")
    print(f"   📉 크기 감소율: {(1 - size_mb_sparse/size_mb_dense)*100:.2f}%")
    print(f"   📊 총 파라미터: {total_params:,} (0이 아닌: {nonzero_params:,})")
    print(f"   ✂️ Sparsity: {sparsity*100:.2f}%")
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
        'model_size_mb': size_mb_sparse,  # Sparse format 크기 사용
        'model_size_mb_dense': size_mb_dense,  # Dense format 크기도 저장
        'memory_usage_mb': memory_usage,
        'meteor_score': avg_meteor,
        'inference_times': latencies,
        'example_caption': example_caption,
        'total_params': total_params,
        'nonzero_params': nonzero_params,
        'sparsity': sparsity,
        'trainable_params': trainable_params,
        'size_reduction': (1 - size_mb_sparse/size_mb_dense)*100 if size_mb_dense > 0 else 0
    }

# ============================================================================
# 시각화
# ============================================================================
def plot_pruning_comparison(results):
    """Pruning 결과 비교 그래프"""
    if not results:
        print("❌ 결과가 없어 plot을 생성할 수 없습니다.")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    precisions = [r['precision'] for r in results]
    mean_times = [r['mean_time_ms'] for r in results]
    std_times = [r['std_time_ms'] for r in results]
    model_sizes = [r['model_size_mb'] for r in results]
    memory_usages = [r['memory_usage_mb'] for r in results]
    meteor_scores = [r.get('meteor_score', None) for r in results]
    sparsities = [r.get('sparsity', 0) * 100 for r in results]
    nonzero_params_list = [r.get('nonzero_params', 0) for r in results]
    
    valid_meteor_scores = [s for s in meteor_scores if s is not None]
    valid_meteor_precisions = [p for p, s in zip(precisions, meteor_scores) if s is not None]
    
    # 색상 설정
    colors = plt.cm.viridis(np.linspace(0, 1, len(precisions)))
    
    # 종합 비교 그래프
    if valid_meteor_scores:
        fig, axes = plt.subplots(3, 2, figsize=(14, 15))
        fig.suptitle('Pruning 성능 비교 종합', fontsize=16, fontweight='bold')
        
        # 1. 추론 시간
        axes[0, 0].bar(precisions, mean_times, alpha=0.8, color=colors, yerr=std_times, capsize=5)
        axes[0, 0].set_ylabel('추론 시간 (ms)', fontweight='bold')
        axes[0, 0].set_title('추론 시간', fontweight='bold')
        axes[0, 0].grid(axis='y', alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        for i, (p, m, s) in enumerate(zip(precisions, mean_times, std_times)):
            axes[0, 0].text(i, m + s + 1, f'{m:.1f}', ha='center', va='bottom', fontsize=9)
        
        # 2. 모델 크기 (Sparse format - 실제 저장 크기)
        axes[0, 1].bar(precisions, model_sizes, alpha=0.8, color=colors, label='Sparse (실제 저장 크기)')
        axes[0, 1].set_ylabel('모델 크기 (MB)', fontweight='bold')
        axes[0, 1].set_title('모델 크기 (Sparse Format)', fontweight='bold')
        axes[0, 1].grid(axis='y', alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].legend()
        for i, (p, s) in enumerate(zip(precisions, model_sizes)):
            axes[0, 1].text(i, s + 0.5, f'{s:.1f}', ha='center', va='bottom', fontsize=9)
        
        # 3. Sparsity
        axes[1, 0].bar(precisions, sparsities, alpha=0.8, color=colors)
        axes[1, 0].set_ylabel('Sparsity (%)', fontweight='bold')
        axes[1, 0].set_title('Sparsity (프루닝 비율)', fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
        for i, (p, s) in enumerate(zip(precisions, sparsities)):
            axes[1, 0].text(i, s + 1, f'{s:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 4. METEOR 점수
        axes[1, 1].bar(valid_meteor_precisions, valid_meteor_scores, alpha=0.8, 
                     color=colors[:len(valid_meteor_scores)])
        axes[1, 1].set_ylabel('METEOR 점수', fontweight='bold')
        axes[1, 1].set_title('METEOR 점수 (캡션 품질)', fontweight='bold')
        axes[1, 1].set_ylim(0, 1.0)
        axes[1, 1].grid(axis='y', alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=45)
        for i, (p, s) in enumerate(zip(valid_meteor_precisions, valid_meteor_scores)):
            axes[1, 1].text(i, s + 0.01, f'{s:.2f}', ha='center', va='bottom', fontsize=9)
        
        # 5. 메모리 사용량
        axes[2, 0].bar(precisions, memory_usages, alpha=0.8, color=colors)
        axes[2, 0].set_ylabel('메모리 사용량 (MB)', fontweight='bold')
        axes[2, 0].set_title('메모리 사용량', fontweight='bold')
        axes[2, 0].grid(axis='y', alpha=0.3)
        axes[2, 0].tick_params(axis='x', rotation=45)
        for i, (p, m) in enumerate(zip(precisions, memory_usages)):
            axes[2, 0].text(i, m + max(memory_usages) * 0.02, f'{m:.1f}', ha='center', va='bottom', fontsize=9)
        
        # 6. 0이 아닌 파라미터 개수
        nonzero_params_m = [p / 1e6 for p in nonzero_params_list]
        axes[2, 1].bar(precisions, nonzero_params_m, alpha=0.8, color=colors)
        axes[2, 1].set_ylabel('0이 아닌 파라미터 (M)', fontweight='bold')
        axes[2, 1].set_title('0이 아닌 파라미터 개수', fontweight='bold')
        axes[2, 1].grid(axis='y', alpha=0.3)
        axes[2, 1].tick_params(axis='x', rotation=45)
        for i, (p, np_m) in enumerate(zip(precisions, nonzero_params_m)):
            axes[2, 1].text(i, np_m + max(nonzero_params_m) * 0.02, f'{np_m:.2f}M', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pruning_comparison_comprehensive.png'), 
                dpi=300, bbox_inches='tight')
    print(f"✅ Plot 저장: {os.path.join(OUTPUT_DIR, 'pruning_comparison_comprehensive.png')}")
    plt.close()

# ============================================================================
# Main
# ============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*70)
    print("=== Pruning 벤치마크 ===")
    print("="*70)
    
    # 1. 모델 및 데이터 로드
    base_model, wm, rwm = load_base_model()
    img_tensor, ref_caption = load_data()
    
    results = []
    
    # 2. 원본 모델 벤치마크 (Baseline)
    print("\n" + "="*70)
    print("=== [Baseline] 원본 모델 ===")
    print("="*70)
    result_baseline = run_benchmark(base_model, img_tensor, wm, rwm, "Original (Baseline)", ref_caption)
    if result_baseline:
        results.append(result_baseline)
    
    # 3. 다양한 Pruning Rate로 테스트
    for pruning_rate in PRUNING_RATES:
        # Magnitude-based Pruning
        print("\n" + "="*70)
        print(f"=== Magnitude Pruning ({pruning_rate*100:.0f}%) ===")
        print("="*70)
        try:
            pruned_model = apply_magnitude_pruning(base_model, pruning_rate)
            pruned_model.to(device)
            result = run_benchmark(
                pruned_model, img_tensor, wm, rwm, 
                f"Magnitude-{pruning_rate*100:.0f}%", ref_caption
            )
            if result:
                results.append(result)
            del pruned_model
            gc.collect()
        except Exception as e:
            print(f"⚠️ Magnitude Pruning ({pruning_rate*100:.0f}%) 실패: {e}")
        
        # Structured Pruning
        print("\n" + "="*70)
        print(f"=== Structured Pruning ({pruning_rate*100:.0f}%) ===")
        print("="*70)
        try:
            pruned_model = apply_structured_pruning(base_model, pruning_rate)
            pruned_model.to(device)
            result = run_benchmark(
                pruned_model, img_tensor, wm, rwm, 
                f"Structured-{pruning_rate*100:.0f}%", ref_caption
            )
            if result:
                results.append(result)
            del pruned_model
            gc.collect()
        except Exception as e:
            print(f"⚠️ Structured Pruning ({pruning_rate*100:.0f}%) 실패: {e}")
    
    # 4. Global Pruning 테스트
    print("\n" + "="*70)
    print("=== Global Pruning (50%) ===")
    print("="*70)
    try:
        pruned_model = apply_global_pruning(base_model, 0.5)
        pruned_model.to(device)
        result = run_benchmark(pruned_model, img_tensor, wm, rwm, "Global-50%", ref_caption)
        if result:
            results.append(result)
        del pruned_model
        gc.collect()
    except Exception as e:
        print(f"⚠️ Global Pruning 실패: {e}")
    
    # 5. 결과 요약 출력
    print("\n" + "="*70)
    print("=== 벤치마크 결과 요약 ===")
    print("="*70)
    if any(r.get('meteor_score') is not None for r in results):
        print(f"{'Method':<25} {'추론시간(ms)':<15} {'모델크기(MB)':<15} {'크기감소(%)':<15} {'Sparsity(%)':<15} {'METEOR':<10}")
        print("-"*100)
        for result in results:
            meteor_str = f"{result.get('meteor_score', 0):.4f}" if result.get('meteor_score') is not None else "N/A"
            sparsity = result.get('sparsity', 0) * 100
            size_reduction = result.get('size_reduction', 0)
            print(f"{result['precision']:<25} "
                  f"{result['mean_time_ms']:.2f}±{result['std_time_ms']:.2f}    "
                  f"{result['model_size_mb']:.2f}          "
                  f"{size_reduction:<15.2f} "
                  f"{sparsity:<15.2f} "
                  f"{meteor_str}")
    else:
        print(f"{'Method':<25} {'추론시간(ms)':<15} {'모델크기(MB)':<15} {'크기감소(%)':<15} {'Sparsity(%)':<15}")
        print("-"*85)
        for result in results:
            sparsity = result.get('sparsity', 0) * 100
            size_reduction = result.get('size_reduction', 0)
            print(f"{result['precision']:<25} "
                  f"{result['mean_time_ms']:.2f}±{result['std_time_ms']:.2f}    "
                  f"{result['model_size_mb']:.2f}          "
                  f"{size_reduction:<15.2f} "
                  f"{sparsity:<15.2f}")
    
    # 6. 시각화
    print("\n" + "="*70)
    print("Plot 생성 중...")
    print("="*70)
    plot_pruning_comparison(results)
    
    print("\n" + "="*70)
    print("=== 벤치마크 완료 ===")
    print(f"결과 저장 위치: {OUTPUT_DIR}")
    print("="*70)

if __name__ == "__main__":
    main()

