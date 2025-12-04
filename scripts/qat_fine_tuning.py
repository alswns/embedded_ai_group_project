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
import platform
import matplotlib.pyplot as plt
from copy import deepcopy
import gc
import warnings

warnings.filterwarnings('ignore')

# 공통 유틸리티 import
from src.utils import (
    setup_device,
    setup_matplotlib,
    get_image_transform,
    count_parameters,
    get_model_size_mb,
    get_peak_memory_mb,
    calculate_meteor,
    CaptionDataset,
    load_test_data,
    prepare_calibration_dataset,
    load_base_model,
    TEST_IMAGE_DIR,
    CAPTIONS_FILE,
)

# ============================================================================
# 설정
# ============================================================================
setup_matplotlib()

OUTPUT_DIR = "qat_results"
QAT_CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
QAT_CHECKPOINT_PATH = os.path.join(QAT_CHECKPOINT_DIR, "qat_checkpoint.pth")
NUM_RUNS = 50

# QAT 설정
QAT_EPOCHS = 30  # QAT 학습 epoch 수 (더 많은 학습으로 더 나은 결과)

# 디바이스 선택
device = setup_device()

# 이미지 전처리
transform = get_image_transform()

# ============================================================================
# 데이터 로드 (공통 모듈 사용)
# ============================================================================
# load_base_model, load_test_data, prepare_calibration_dataset는 utils에서 import

# ============================================================================
# 체크포인트 관리
# ============================================================================
def save_qat_checkpoint(model, optimizer, epoch, loss_history, word_map, checkpoint_path):
    """QAT 체크포인트 저장"""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_history': loss_history,
        'word_map': word_map,
        'qat_epochs': QAT_EPOCHS,
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"   💾 체크포인트 저장: {checkpoint_path} (Epoch {epoch})")

def load_qat_checkpoint(checkpoint_path, model, optimizer=None):
    """QAT 체크포인트 로드"""
    if not os.path.exists(checkpoint_path):
        return None, None, 0, []
    
    print(f"   📂 체크포인트 로드 중: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    start_epoch = checkpoint.get('epoch', 0)
    loss_history = checkpoint.get('loss_history', [])
    word_map = checkpoint.get('word_map', None)
    
    # 모델 상태 로드
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"   ✅ 모델 상태 로드 완료 (Epoch {start_epoch})")
    except Exception as e:
        print(f"   ⚠️ 모델 상태 로드 실패: {e}")
        return None, word_map, 0, []
    
    # Optimizer 상태 로드
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"   ✅ Optimizer 상태 로드 완료")
        except Exception as e:
            print(f"   ⚠️ Optimizer 상태 로드 실패: {e}")
    
    return model, word_map, start_epoch, loss_history

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
    cal_images, _ = prepare_calibration_dataset(word_map, num_samples=1000, transform=transform)
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

    # 체크포인트 확인
    checkpoint_exists = os.path.exists(QAT_CHECKPOINT_PATH)
    
    if not checkpoint_exists:
        # 체크포인트가 없으면 양자화 준비 수행
        print("   📊 Calibration 데이터 준비 중...")
        cal_images, _ = prepare_calibration_dataset(word_map, num_samples=1000, transform=transform)
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
    else:
        print("   📂 체크포인트 발견 - 양자화 준비 단계 건너뜀")
        model_cpu.train()
    
    # 학습 데이터셋 준비
    try:
        from torch.utils.data import DataLoader
        
        MAX_CAPTION_LEN = 50
        
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
            batch_size=64, 
            shuffle=True, 
            num_workers=0,
            pin_memory=False
        )
        
        print(f"   📚 학습 데이터: {len(dataset)}개 샘플")
        
        # Mixed Precision 설정
        # QAT는 양자화 연산을 포함하므로 CPU에서만 안정적으로 동작
        # MPS는 양자화 연산(aten::_fused_moving_avg_obs_fq_helper)을 지원하지 않음
        use_mixed_precision = False
        scaler = None
        qat_device = torch.device("cpu")
        
        if torch.cuda.is_available():
            # CUDA는 양자화를 지원하지만, 안정성을 위해 CPU 사용 권장
            # 필요시 아래 주석을 해제하여 CUDA 사용 가능
            # qat_device = torch.device("cuda")
            # model_cpu = model_cpu.to(qat_device)
            # use_mixed_precision = True
            # scaler = torch.cuda.amp.GradScaler()
            # print("   🚀 GPU 사용 - FP16 Mixed Precision 활성화")
            print("   💻 CPU 사용 - QAT는 CPU에서 안정적으로 동작 (양자화 연산 지원)")
        else:
            print("   💻 CPU 사용 - FP32 학습")
        
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = torch.optim.Adam(model_cpu.parameters(), lr=1e-4)
        vocab_size = len(word_map)
        
        # 체크포인트 로드 시도
        start_epoch = 0
        loss_history = []
        
        if checkpoint_exists:
            loaded_model, loaded_word_map, loaded_epoch, loaded_loss_history = load_qat_checkpoint(
                QAT_CHECKPOINT_PATH, model_cpu, optimizer
            )
            
            if loaded_model is not None:
                model_cpu = loaded_model
                if loaded_word_map:
                    word_map = loaded_word_map
                start_epoch = loaded_epoch
                loss_history = loaded_loss_history
                
                if start_epoch >= qat_epochs:
                    print(f"   ✅ 학습이 이미 완료되었습니다 (Epoch {start_epoch}/{qat_epochs})")
                    print("   🔄 양자화 변환 진행...")
                else:
                    print(f"   🔄 체크포인트에서 이어서 학습: Epoch {start_epoch + 1}/{qat_epochs}부터 시작")
            else:
                print(f"   ⚠️ 체크포인트 로드 실패 - 새로운 학습 시작")
        else:
            print(f"   🆕 새로운 학습 시작: {qat_epochs} epochs")
        
        # 학습 루프 (학습이 완료되지 않은 경우에만)
        if start_epoch < qat_epochs:
            for epoch in range(start_epoch, qat_epochs):
                epoch_loss = 0
                num_batches = 0
                
                for batch_idx, (imgs, caps) in enumerate(dataloader):
                    # if batch_idx >= 30:  # 더 많은 배치로 학습
                    #     break
                    
                    imgs = imgs.to(qat_device)
                    caps = caps.to(qat_device)
                    
                    optimizer.zero_grad()
                    
                    try:
                        # QAT는 CPU에서만 수행 (양자화 연산이 MPS에서 지원되지 않음)
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
                    loss_history.append(avg_loss)
                    print(f"      Epoch {epoch+1}/{qat_epochs}, Loss: {avg_loss:.4f}")
                    
                    # 체크포인트 저장 (매 epoch마다)
                    save_qat_checkpoint(
                        model_cpu, optimizer, epoch + 1, loss_history, word_map, QAT_CHECKPOINT_PATH
                    )
        else:
            print(f"   ⏭️ 학습 완료 - 양자화 변환으로 진행")
        
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
        
        # 최종 양자화 모델 저장
        final_model_path = os.path.join(QAT_CHECKPOINT_DIR, "qat_final_model.pth")
        os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
        torch.save({
            'model_state_dict': quantized_model.state_dict(),
            'word_map': word_map,
            'loss_history': loss_history,
            'qat_epochs': qat_epochs,
            'final_epoch': qat_epochs,
        }, final_model_path)
        print(f"   💾 최종 양자화 모델 저장: {final_model_path}")
        
        # 최종 체크포인트 저장 (학습 완료 표시)
        save_qat_checkpoint(
            model_cpu, optimizer, qat_epochs, loss_history, word_map, QAT_CHECKPOINT_PATH
        )
        
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
    base_model, wm, rwm = load_base_model(device=device)
    img_tensor, ref_caption = load_test_data(device=device, transform=transform)
    
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

