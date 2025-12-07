"""
Jetson Nano 모델 성능 비교 벤치마크
- Base Model (원본)
- Base Model + FP16 양자화
- Base Model + Pruning
- Base Model + Pruning + Fine-tuning
- Base Model + Pruning + Fine-tuning + FP16
"""

import os
import gc
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import psutil
import sys
from pathlib import Path

# 프로젝트 모듈
from src.utils.memory_safe_import import load_model_class
from src.utils.model_utils import get_model_size_mb

# FLOPs 계산
try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("⚠️  thop 라이브러리 없음 (pip install thop)", file=sys.stderr)

# ============================================================================
# 설정
# ============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_RUNS = 50  # 각 모델당 추론 횟수
WARMUP_RUNS = 5  # 워밍업 횟수

print("📊 Jetson Nano 모델 비교 벤치마크")
print("=" * 70)
print("디바이스: {}".format(device))
print("=" * 70 + "\n")

# ============================================================================
# 모델 경로 설정
# ============================================================================
MODELS_CONFIG = {
    'Base Model': {
        'path': 'models/lightweight_captioning_model.pth',
        'quantize': False,
        'pruned': False,
        'finetuned': False,
    },
    'Base Model + FP16': {
        'path': 'models/lightweight_captioning_model.pth',
        'quantize': True,
        'pruned': False,
        'finetuned': False,
    },
    'Base Model + Pruning': {
        'path': 'pruning_results/Pruning_epoch_1_checkpoint.pt',
        'quantize': False,
        'pruned': True,
        'finetuned': False,
    },
    'Base Model + Pruning + FT': {
        'path': 'pruning_results/Pruning_epoch_1_checkpoint.pt',
        'quantize': False,
        'pruned': True,
        'finetuned': True,
    },
    'Base Model + Pruning + FT + FP16': {
        'path': 'pruning_results/Pruning_epoch_1_checkpoint.pt',
        'quantize': True,
        'pruned': True,
        'finetuned': True,
    },
}

# ============================================================================
# 모델 로드 함수
# ============================================================================
def load_model_with_config(config):
    """설정에 따라 모델 로드"""
    model_path = config['path']
    
    if not os.path.exists(model_path):
        print("❌ 모델 파일 없음: {}".format(model_path))
        return None, None, None
    
    try:
        # 모델 클래스 로드
        Model = load_model_class()
        
        # 체크포인트 로드
        checkpoint = torch.load(model_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            word_map = checkpoint.get('word_map')
            rev_word_map = checkpoint.get('rev_word_map')
            vocab_size = checkpoint.get('vocab_size')
            state_dict = checkpoint['model_state_dict']
            
            # state_dict에서 모델 크기 추출
            decoder_dim = checkpoint.get('decoder_dim', 512)
            attention_dim = checkpoint.get('attention_dim', 256)
            
            # state_dict에서 실제 크기 추출
            if 'decoder.decode_step.weight_ih' in state_dict:
                actual_size = state_dict['decoder.decode_step.weight_ih'].shape[0]
                decoder_dim = actual_size // 3
            
            if 'decoder.encoder_att.weight' in state_dict:
                attention_dim = state_dict['decoder.encoder_att.weight'].shape[0]
            
            # 모델 생성
            gc.collect()
            model = Model(
                vocab_size=vocab_size,
                embed_dim=300,
                decoder_dim=decoder_dim,
                attention_dim=attention_dim
            )
            del Model
            
            # 가중치 로드
            try:
                model.load_state_dict(state_dict, strict=True)
            except:
                model.load_state_dict(state_dict, strict=False)
            
            model = model.to(device)
            model.eval()
            
            # ★ FP16 양자화 적용
            if config['quantize']:
                model = model.half()
            
            return model, word_map, rev_word_map
        else:
            print("❌ 잘못된 모델 파일 형식")
            return None, None, None
            
    except Exception as e:
        print("❌ 모델 로드 실패: {}".format(e))
        import traceback
        traceback.print_exc()
        return None, None, None

# ============================================================================
# FLOPs 계산 함수
# ============================================================================
def calculate_flops(model):
    """
    모델의 FLOPs 계산 (간단한 추정 방식)
    thop 라이브러리 사용 불가시 파라미터 기반 추정
    """
    try:
        # thop 라이브러리로 정확한 FLOPs 계산 시도
        if THOP_AVAILABLE:
            try:
                dummy_input = torch.randn(1, 3, 224, 224).to(device)
                if hasattr(model, 'float'):
                    model_float = model.float()
                    dummy_input_float = dummy_input.float()
                    from thop import profile
                    flops, params = profile(model_float, inputs=(dummy_input_float,), verbose=False)
                else:
                    from thop import profile
                    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
                
                return float(flops / 1e6)  # Millions
            except Exception as e:
                print("⚠️  thop 계산 실패: {}".format(e))
    except:
        pass
    
    # 폴백: 파라미터 개수 기반 추정
    # 일반적으로 FLOPs ≈ 2 × Parameters (순전파)
    param_count = sum(p.numel() for p in model.parameters())
    estimated_flops = param_count * 2.5 / 1e6  # Millions
    
    return estimated_flops

# ============================================================================
# 성능 측정 함수
# ============================================================================
class BenchmarkMetrics:
    """모델 성능 메트릭"""
    def __init__(self):
        self.inference_times = []
        self.memory_usage = []
        self.process = psutil.Process(os.getpid())
    
    def record_inference(self, inf_time):
        """추론 시간 기록"""
        self.inference_times.append(inf_time)
    
    def record_memory(self):
        """메모리 기록"""
        mem = self.process.memory_info().rss / 1024 / 1024
        self.memory_usage.append(mem)
    
    def get_stats(self):
        """통계 계산"""
        if not self.inference_times:
            return None
        
        times = np.array(self.inference_times)
        
        return {
            'mean_latency_ms': float(np.mean(times)),
            'median_latency_ms': float(np.median(times)),
            'min_latency_ms': float(np.min(times)),
            'max_latency_ms': float(np.max(times)),
            'std_latency_ms': float(np.std(times)),
            'fps': float(1000.0 / np.mean(times)),
            'cpu_memory_mb': float(np.mean(self.memory_usage) if self.memory_usage else 0),
            'total_params': 0,
            'model_size_mb': 0,
            'flops_millions': 0,  # 추후 계산
        }

def benchmark_model(model, word_map, rev_word_map, model_name, config):
    """단일 모델 벤치마크"""
    print("\n🔍 벤치마킹: {}".format(model_name))
    print("-" * 70)
    
    metrics = BenchmarkMetrics()
    
    # 더미 입력 생성
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    if config['quantize']:
        dummy_input = dummy_input.half()
    
    # 워밍업
    print("  ⚙️  워밍업 중...", end='')
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            _ = model.encoder(dummy_input)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    print(" ✅")
    
    # 본 벤치마크
    print("  🏃 추론 실행 중... ({} runs)".format(NUM_RUNS))
    
    # 더미 이미지 생성
    dummy_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    
    with torch.no_grad():
        for i in range(NUM_RUNS):
            # 전처리
            img = dummy_frame.copy()
            img = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)
            
            if config['quantize']:
                img_tensor = img_tensor.half()
            
            # 추론
            metrics.record_memory()
            start = time.time()
            
            try:
                _ = model.generate(img_tensor, word_map, rev_word_map, max_len=50, device=device)
            except:
                # generate가 없으면 encoder만 실행
                features = model.encoder(img_tensor)
                if hasattr(model, 'decoder'):
                    batch_size = features.size(0)
                    channel = features.size(1)
                    features_flat = features.view(batch_size, channel, -1).permute(0, 2, 1)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            inference_time = (time.time() - start) * 1000
            metrics.record_inference(inference_time)
            
            if (i + 1) % 10 == 0:
                print("    [{}/{}] 완료".format(i + 1, NUM_RUNS))
    
    # 통계 계산
    stats = metrics.get_stats()
    
    # 모델 크기 정보
    param_count = sum(p.numel() for p in model.parameters())
    stats['total_params'] = param_count
    stats['model_size_mb'] = get_model_size_mb(model)
    
    # ★ FLOPs 계산
    flops_millions = calculate_flops(model)
    stats['flops_millions'] = flops_millions
    
    print("\n  📊 결과:")
    print("    • 평균 지연시간: {:.2f} ms".format(stats['mean_latency_ms']))
    print("    • FPS: {:.1f}".format(stats['fps']))
    print("    • 메모리: {:.1f} MB".format(stats['cpu_memory_mb']))
    print("    • 모델 크기: {:.2f} MB".format(stats['model_size_mb']))
    print("    • 파라미터: {:,}개".format(param_count))
    print("    • FLOPs: {:.1f}M".format(flops_millions))
    
    return stats

# ============================================================================
# 메인 벤치마크 실행
# ============================================================================
def main():
    results = {}
    
    print("\n" + "=" * 70)
    print("🚀 모델 성능 비교 시작")
    print("=" * 70)
    
    for model_name, config in MODELS_CONFIG.items():
        print("\n\n[{}/{}] {}".format(
            list(MODELS_CONFIG.keys()).index(model_name) + 1,
            len(MODELS_CONFIG),
            model_name
        ))
        
        # 모델 로드
        model, word_map, rev_word_map = load_model_with_config(config)
        
        if model is None:
            print("⚠️  모델 로드 실패, 스킵")
            continue
        
        try:
            # 벤치마크 실행
            stats = benchmark_model(model, word_map, rev_word_map, model_name, config)
            results[model_name] = stats
            
        except Exception as e:
            print("❌ 벤치마크 실패: {}".format(e))
            import traceback
            traceback.print_exc()
        
        finally:
            # 메모리 정리
            del model
            gc.collect()
    
    # 결과 요약
    print("\n\n" + "=" * 70)
    print("📊 벤치마크 결과 요약")
    print("=" * 70)
    
    for model_name, stats in results.items():
        print("\n{}:".format(model_name))
        print("  • Latency: {:.2f} ms".format(stats['mean_latency_ms']))
        print("  • FPS: {:.1f}".format(stats['fps']))
        print("  • Memory: {:.1f} MB".format(stats['cpu_memory_mb']))
        print("  • Model Size: {:.2f} MB".format(stats['model_size_mb']))
        print("  • FLOPs: {:.1f}M".format(stats['flops_millions']))
    
    # 그래프 생성
    print("\n\n📈 그래프 생성 중...")
    plot_comparison(results)
    
    # 결과 저장
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("✅ 결과 저장: benchmark_results.json")

# ============================================================================
# 그래프 생성 함수
# ============================================================================
def plot_comparison(results):
    """모델 성능 비교 그래프"""
    if not results:
        print("❌ 결과가 없습니다")
        return
    
    model_names = list(results.keys())
    
    # 데이터 추출
    latencies = [results[m]['mean_latency_ms'] for m in model_names]
    fps_values = [results[m]['fps'] for m in model_names]
    memory_usage = [results[m]['cpu_memory_mb'] for m in model_names]
    model_sizes = [results[m]['model_size_mb'] for m in model_names]
    param_counts = [results[m]['total_params'] / 1e6 for m in model_names]  # Million
    flops_values = [results[m]['flops_millions'] for m in model_names]  # ★ FLOPs 추가
    
    # 그래프 생성 (3x3으로 변경하여 7개 그래프 표시)
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('Jetson Nano 모델 성능 비교 분석 (FLOPs 포함)', fontsize=16, fontweight='bold')
    
    # 색상 설정
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # 1. 추론 지연시간
    axes[0, 0].bar(range(len(model_names)), latencies, color=colors, alpha=0.8)
    axes[0, 0].set_ylabel('지연시간 (ms)', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('① 평균 추론 지연시간', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks(range(len(model_names)))
    axes[0, 0].set_xticklabels([m.replace(' + ', '\n+ ') for m in model_names], fontsize=9)
    axes[0, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(latencies):
        axes[0, 0].text(i, v + 1, '{:.1f}ms'.format(v), ha='center', fontsize=10, fontweight='bold')
    
    # 2. FPS (처리량)
    axes[0, 1].bar(range(len(model_names)), fps_values, color=colors, alpha=0.8)
    axes[0, 1].set_ylabel('FPS', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('② 초당 프레임 수 (FPS)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks(range(len(model_names)))
    axes[0, 1].set_xticklabels([m.replace(' + ', '\n+ ') for m in model_names], fontsize=9)
    axes[0, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(fps_values):
        axes[0, 1].text(i, v + 0.5, '{:.1f}'.format(v), ha='center', fontsize=10, fontweight='bold')
    
    # 3. CPU 메모리 사용량
    axes[0, 2].bar(range(len(model_names)), memory_usage, color=colors, alpha=0.8)
    axes[0, 2].set_ylabel('메모리 (MB)', fontsize=11, fontweight='bold')
    axes[0, 2].set_title('③ CPU 메모리 사용량', fontsize=12, fontweight='bold')
    axes[0, 2].set_xticks(range(len(model_names)))
    axes[0, 2].set_xticklabels([m.replace(' + ', '\n+ ') for m in model_names], fontsize=9)
    axes[0, 2].grid(axis='y', alpha=0.3)
    for i, v in enumerate(memory_usage):
        axes[0, 2].text(i, v + 10, '{:.0f}MB'.format(v), ha='center', fontsize=10, fontweight='bold')
    
    # 4. 모델 크기
    axes[1, 0].bar(range(len(model_names)), model_sizes, color=colors, alpha=0.8)
    axes[1, 0].set_ylabel('크기 (MB)', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('④ 모델 파일 크기', fontsize=12, fontweight='bold')
    axes[1, 0].set_xticks(range(len(model_names)))
    axes[1, 0].set_xticklabels([m.replace(' + ', '\n+ ') for m in model_names], fontsize=9)
    axes[1, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(model_sizes):
        axes[1, 0].text(i, v + 0.5, '{:.2f}MB'.format(v), ha='center', fontsize=10, fontweight='bold')
    
    # 5. 파라미터 개수
    axes[1, 1].bar(range(len(model_names)), param_counts, color=colors, alpha=0.8)
    axes[1, 1].set_ylabel('파라미터 (Million)', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('⑤ 총 파라미터 개수', fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks(range(len(model_names)))
    axes[1, 1].set_xticklabels([m.replace(' + ', '\n+ ') for m in model_names], fontsize=9)
    axes[1, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(param_counts):
        axes[1, 1].text(i, v + 0.1, '{:.1f}M'.format(v), ha='center', fontsize=10, fontweight='bold')
    
    # ★ 6. FLOPs (새로 추가)
    axes[1, 2].bar(range(len(model_names)), flops_values, color=colors, alpha=0.8)
    axes[1, 2].set_ylabel('FLOPs (Million)', fontsize=11, fontweight='bold')
    axes[1, 2].set_title('⑥ 부동소수점 연산수 (FLOPs)', fontsize=12, fontweight='bold')
    axes[1, 2].set_xticks(range(len(model_names)))
    axes[1, 2].set_xticklabels([m.replace(' + ', '\n+ ') for m in model_names], fontsize=9)
    axes[1, 2].grid(axis='y', alpha=0.3)
    for i, v in enumerate(flops_values):
        axes[1, 2].text(i, v + 50, '{:.0f}M'.format(v), ha='center', fontsize=10, fontweight='bold')
    
    # 7. 성능-메모리 트레이드오프 (Latency × Memory)
    tradeoff = [lat * mem for lat, mem in zip(latencies, memory_usage)]
    axes[2, 0].bar(range(len(model_names)), tradeoff, color=colors, alpha=0.8)
    axes[2, 0].set_ylabel('트레이드오프 (ms×MB)', fontsize=11, fontweight='bold')
    axes[2, 0].set_title('⑦ 성능-메모리 트레이드오프\n(낮을수록 우수)', fontsize=12, fontweight='bold')
    axes[2, 0].set_xticks(range(len(model_names)))
    axes[2, 0].set_xticklabels([m.replace(' + ', '\n+ ') for m in model_names], fontsize=9)
    axes[2, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(tradeoff):
        axes[2, 0].text(i, v + 10, '{:.0f}'.format(v), ha='center', fontsize=10, fontweight='bold')
    
    # 나머지 빈 서브플롯 숨기기
    axes[2, 1].axis('off')
    axes[2, 2].axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 저장
    output_path = 'benchmark_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print("✅ 그래프 저장: {}".format(output_path))
    
    # 상세 테이블
    plot_comparison_table(results)
    
    plt.show()

def plot_comparison_table(results):
    """상세 비교 테이블"""
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('off')
    
    # 테이블 데이터
    model_names = list(results.keys())
    table_data = []
    
    for model_name in model_names:
        stats = results[model_name]
        table_data.append([
            model_name,
            '{:.2f}ms'.format(stats['mean_latency_ms']),
            '{:.1f}'.format(stats['fps']),
            '{:.1f}MB'.format(stats['cpu_memory_mb']),
            '{:.2f}MB'.format(stats['model_size_mb']),
            '{:.1f}M'.format(stats['total_params'] / 1e6),
            '{:.0f}M'.format(stats['flops_millions']),  # ★ FLOPs 추가
        ])
    
    # 테이블 생성
    table = ax.table(
        cellText=table_data,
        colLabels=['모델', '지연시간', 'FPS', 'CPU 메모리', '모델 크기', '파라미터', 'FLOPs'],
        cellLoc='center',
        loc='center',
        colWidths=[0.22, 0.12, 0.10, 0.13, 0.13, 0.12, 0.12]  # ★ FLOPs 추가
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # 헤더 스타일
    for i in range(len(table_data[0]) + 1):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 행 색상
    colors = ['#E8F5E9', '#FFF3E0', '#E3F2FD', '#FCE4EC', '#F3E5F5']
    for i in range(len(table_data)):
        for j in range(len(table_data[0]) + 1):
            table[(i + 1, j)].set_facecolor(colors[i % len(colors)])
    
    plt.title('Jetson Nano 모델 성능 비교 상세표', fontsize=14, fontweight='bold', pad=20)
    plt.savefig('benchmark_comparison_table.png', dpi=150, bbox_inches='tight')
    print("✅ 상세표 저장: benchmark_comparison_table.png")

if __name__ == "__main__":
    main()
