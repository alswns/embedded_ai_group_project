"""
Pruning 벤치마크 스크립트 (Jetson Nano 최적화)
다양한 Pruning 기법을 적용하고 성능을 비교합니다.
"""
import os
import gc
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import multiprocessing

warnings.filterwarnings('ignore')

# ★ Jetson Nano 최적화
np.seterr(over='ignore', under='ignore')

# 공통 유틸리티 import
from src.utils import (
    # Config
    setup_device,
    setup_matplotlib,
    get_image_transform,
    TEST_IMAGE_DIR,
    CAPTIONS_FILE,
    # Dataset
    CaptionDataset,
    load_test_data,
    load_base_model,
    # Benchmark
    clear_memory,
    calculate_meteor,
    run_benchmark,
    # Pruning
    count_nonzero_parameters,
    apply_magnitude_pruning,
    apply_structured_pruning,
    # Finetune
    load_checkpoint,
    setup_training,
    save_checkpoint,
    print_checkpoint_info,
    restore_optimizer,
    apply_magnitude_mask,
    # Quantization
    apply_dynamic_quantization,
)

# ============================================================================
# 설정
# ============================================================================
setup_matplotlib()

OUTPUT_DIR = "pruning_results"
NUM_RUNS = 50

# Pruning 설정
PRUNING_RATES = [0.1, 0.3, 0.5, 0.7, 0.9]

ENABLE_MAGNITUDE_PRUNING = True
ENABLE_FINETUNING = True
FINETUNE_RATES = [0.3]
#31
FINETUNE_EPOCHS = 16
LEARNING_RATE = 5e-5
EARLY_STOPPING_PATIENCE = 5
MAX_PRUNING_RATE = 0.51
METEO_IMAGE_NUM = 100
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

# Quantization 설정
ENABLE_QUANTIZATION = True
QUANTIZATION_METHOD = 'dynamic'  # 'dynamic', 'static', 'qat'

# 디바이스 선택
# ★ Jetson Nano 최적화: CPU 성능 최대화
num_cores = multiprocessing.cpu_count()
optimal_threads = max(2, num_cores - 1)
torch.set_num_threads(optimal_threads)
torch.set_num_interop_threads(2)

import cv2
cv2.setNumThreads(0)  # OpenCV 병렬화 비활성화

device = setup_device()
if ENABLE_QUANTIZATION:
    device = torch.device('cpu')

print("📍 Jetson Nano 최적화 설정: {} 스레드".format(optimal_threads))

transform = get_image_transform()


# ============================================================================
# 벤치마크 래퍼
# ============================================================================
def quantize_benchmark(model, img_tensor, wm, rwm, ref_caption, baseline_params, results, label, val_dataloader=None):
    q_model = apply_dynamic_quantization(model)
    
    result = run_benchmark(q_model, img_tensor, wm, rwm, "{} Quantization".format(label),ref_caption=ref_caption,
    baseline_params=baseline_params,
    num_runs=NUM_RUNS,
    num_meteor_images=METEO_IMAGE_NUM,
    val_dataloader=val_dataloader,
    transform=transform,
    calculate_meteor_fn=calculate_meteor)

    if result is not None:
        results.append(result)
    del q_model

def run_pruning_benchmark(pruned_model, label, img_tensor, wm, rwm, ref_caption, 
                         baseline_params, device, results, val_dataloader=None):
    """프루닝된 모델 벤치마크 및 파인튜닝 실행"""
    
    pruned_model.to(device)

    # 프루닝 후 벤치마크
    result = run_benchmark(
        pruned_model, img_tensor, wm, rwm, label,
        ref_caption=ref_caption,
        baseline_params=baseline_params,
        num_runs=NUM_RUNS,
        num_meteor_images=METEO_IMAGE_NUM,
        val_dataloader=val_dataloader,
        transform=transform,
        calculate_meteor_fn=calculate_meteor
    )
    if result:
        results.append(result)

    if ENABLE_QUANTIZATION:
        quantize_benchmark(pruned_model, img_tensor, wm, rwm, ref_caption, baseline_params, results, label, val_dataloader=val_dataloader)

    if ENABLE_FINETUNING:
        fine_tuned_model = fine_tune_pruned_model(
            pruned_model, wm, 
            img_tensor=img_tensor, wm=wm, rwm=rwm,
            ref_caption=ref_caption, baseline_params=baseline_params,
            epochs=FINETUNE_EPOCHS, label=label.replace(" ", "_").replace("%", "pct"),
            learning_rate=LEARNING_RATE
        )
        fine_tuned_model.to(device)
        
        # 파인 튜닝 후 최종 벤치마크
        result_finetuned = run_benchmark(
                    fine_tuned_model, img_tensor, wm, rwm,
                    "Fine-tuned",
                    ref_caption=ref_caption,
                    baseline_params=baseline_params,
                    num_runs=NUM_RUNS,
                    num_meteor_images=METEO_IMAGE_NUM,
                    val_dataloader=val_dataloader,
                    transform=transform,
                    calculate_meteor_fn=calculate_meteor
                )
        if result_finetuned:
            results.append(result_finetuned)

        # 양자화 적용 (선택적)
            if ENABLE_QUANTIZATION:
                quantize_benchmark(fine_tuned_model, img_tensor, wm, rwm, ref_caption, baseline_params, results, "Pruning Fine-tuned", val_dataloader=val_dataloader)
        
            
        del fine_tuned_model
    
    
    gc.collect()
# ============================================================================
# 파인튜닝 함수
# ============================================================================
def getDataset(word_map):
    full_dataset = CaptionDataset(
            images_dir=TEST_IMAGE_DIR,
            captions_file=CAPTIONS_FILE,
            transform=transform,
            word_map=word_map,
            max_len=50
    )
    if len(full_dataset)==0:
        return None, None
    # 학습/검증 분할
    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    
    print("   📊 데이터셋: 학습({}개) / 검증({}개)".format(train_size, val_size))
    
    batch_size = 32 if train_size < 1000 else 64
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_dataloader, val_dataloader

def fine_tune_pruned_model(model, word_map, img_tensor=None, wm=None, rwm=None, 
                          ref_caption=None, baseline_params=None, epochs=2, 
                          label="pruned_model", learning_rate=5e-5, val_dataloader=None):
    """파인튜닝 수행"""
    print("\n   🔄 파인 튜닝 시작 ({} epoch)...".format(epochs))
    
    # 마스크 확인
    if hasattr(model, '_magnitude_pruning_masks'):
        masks = model._magnitude_pruning_masks
        print("   ✅ {}개 마스크 감지".format(len(masks)))
    print(label)
    # 체크포인트 로드
    checkpoint, start_epoch, checkpoint_path = load_checkpoint(label, device)
    optimizer_state = checkpoint.get('optimizer_state_dict') if checkpoint else None
    
    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print_checkpoint_info(checkpoint, start_epoch)
        print("   ✅ Epoch {}부터 재개합니다.".format(start_epoch+1))
    else:
        print("   ℹ️ 처음부터 시작합니다.")

    # 학습 설정
    optimizer, criterion = setup_training(model, learning_rate, device)
    restore_optimizer(optimizer, optimizer_state)
    
    # 데이터셋 준비
    try:
        train_dataloader, val_dataloader = getDataset(word_map)
        if train_dataloader is None or val_dataloader is None:
            print("   ⚠️ 데이터셋이 없어 파인 튜닝을 건너뜁니다.")
            return model
        
        model.train()
        model.to(device)
        
        vocab_size = len(word_map)
        rev_word_map = {v: k for k, v in word_map.items()}
        
        best_meteor_score = -float('in')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(start_epoch, epochs):
            print("   🏋️ Epoch {epoch+1}/{epochs}")
            total_loss = 0
            num_batches = 0
            
            train_iter = tqdm(enumerate(train_dataloader), total=len(train_dataloader), 
                             desc="      학습 중", ncols=100)
            
            for batch_idx, (imgs, caps) in train_iter:
                imgs = imgs.to(device)
                caps = caps.to(device)
                
                optimizer.zero_grad()
                
                try:
                    outputs, alphas = model(imgs, caps)
                    targets = caps[:, 1:]
                    outputs = outputs[:, :targets.shape[1], :]
                    loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
                    loss.backward()
                    optimizer.step()
                    
                    # 마스크 강제 적용
                    apply_magnitude_mask(model)
                    
                    total_loss += loss.item()
                    num_batches += 1
                except Exception as e:
                    continue
                
                if (batch_idx + 1) % 10 == 0:
                    train_iter.set_postfix(loss="{:.4f}".format(total_loss / num_batches))
            
            # Epoch 완료
            avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
            print("   ✅ Epoch {} 완료 (학습 Loss: {:.4f})".format(epoch+1, avg_loss))
            
            # 검증
            model.eval()
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for val_imgs, val_caps in tqdm(val_dataloader, desc="      검증 중", ncols=100):
                    val_imgs = val_imgs.to(device)
                    val_caps = val_caps.to(device)
                    
                    try:
                        val_outputs, _ = model(val_imgs, val_caps)
                        val_targets = val_caps[:, 1:]
                        val_outputs = val_outputs[:, :val_targets.shape[1], :]
                        val_loss_batch = criterion(val_outputs.reshape(-1, vocab_size), val_targets.reshape(-1))
                        val_loss += val_loss_batch.item()
                        val_batches += 1
                    except:
                        continue
            
            avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
            print("      검증 Loss: {:.4f}".format(avg_val_loss))
            
            model.train()
            
            # 벤치마크
            current_meteor_score = None
            if img_tensor is not None and wm is not None and rwm is not None:
                print("\n   📊 Epoch {} 벤치마크...".format(epoch+1))
                model.eval()
                benchmark_result = run_benchmark(
                    model, img_tensor, wm, rwm,
                    "Fine-tuned (Epoch {}/{})".format(epoch+1, epochs),
                    ref_caption=ref_caption,
                    baseline_params=baseline_params,
                    num_runs=NUM_RUNS,
                    num_meteor_images=METEO_IMAGE_NUM,
                    val_dataloader=val_dataloader,
                    transform=transform,
                    calculate_meteor_fn=calculate_meteor
                )
                model.train()
                
            # Early Stopping
            if current_meteor_score is not None:
                if current_meteor_score > best_meteor_score:
                    best_meteor_score = current_meteor_score
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                    print("   🎉 최고 METEOR: {:.4f}".format(best_meteor_score))
                elif val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                    print("   🎉 새로운 최저 검증 Loss: {:.4f}".format(best_loss))
                else:
                    patience_counter += 1
                    print("   ⚠️ METEOR 미개선 (Patience: {}/{})".format(patience_counter, EARLY_STOPPING_PATIENCE))
                    
                    if patience_counter >= EARLY_STOPPING_PATIENCE:
                        print("\n   🛑 Early Stopping!")
                        if best_model_state:
                            model.load_state_dict(best_model_state)
                        break
                
                # 체크포인트 저장
                save_checkpoint(
                    model, optimizer, epoch, label,
                    word_map=word_map,
                    rev_word_map=rev_word_map,
                    vocab_size=vocab_size,
                    avg_train_loss=avg_loss,
                    avg_val_loss=avg_val_loss,
                    meteor_score=current_meteor_score
                )
        
        model.eval()
        return model
        
    except Exception as e:
        print("   ⚠️ 파인 튜닝 실패: {}".format(e))
        import traceback
        traceback.print_exc()
        return model


# ============================================================================
# 시각화
# ============================================================================
def plot_embedded_metrics(results):
    """임베디드 환경 특화 지표 종합 분석"""
    if not results:
        print("❌ 결과가 없어 plot을 생성할 수 없습니다.")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 파인 튜닝된 결과 제외
    precisions = [r['precision'] for r in results]
    mean_times = [r['mean_time_ms'] for r in results]
    model_sizes = [r['model_size_mb'] for r in results]
    model_memory = [r.get('model_memory_mb', r['model_size_mb']) for r in results]
    inference_memory = [r.get('inference_memory_mb', 0) for r in results]
    ms_per_token = [r.get('mean_ms_per_token', r['mean_time_ms']/10) for r in results]
    total_params = [r.get('total_params', 0) / 1e6 for r in results]
    meteor_scores = [r.get('meteor_score', None) for r in results]
    
    # 추가 지표 계산
    baseline_params = results[0]['total_params']
    baseline_nonzero = results[0]['nonzero_params']
    flops_reduction = []
    for r in results:
        if r['sparsity'] > 1.0 and r['total_params'] == baseline_params:
            reduction = (1 - r['nonzero_params'] / baseline_nonzero) * 100
        else:
            reduction = (1 - r['total_params'] / baseline_params) * 100
        flops_reduction.append(reduction)
    
    baseline_size = results[0]['model_size_mb']
    size_reduction = [(1 - s / baseline_size) * 100 for s in model_sizes]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(precisions)))
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('임베디드 환경 최적화 종합 분석', fontsize=16, fontweight='bold')
    
    # 1. 토큰당 추론 시간
    bar=axes[0, 0].bar(precisions, ms_per_token, alpha=0.8, color=colors)
    for rect, ms in zip(bar, ms_per_token):
        height = rect.get_height()
        axes[0, 0].text(rect.get_x() + rect.get_width() / 2.0, height, '{:.1f}'.format(ms), 
                        ha='center', va='bottom', fontsize=8)
    axes[0, 0].set_ylabel('시간 (ms/token)')
    axes[0, 0].set_title('① 토큰당 추론 시간')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # 2. 모델 크기
    bar=axes[0, 1].bar(precisions, model_sizes, alpha=0.8, color=colors)
    for rect, size in zip(bar, model_sizes):
        height = rect.get_height()
        axes[0, 1].text(rect.get_x() + rect.get_width() / 2.0, height, '{:.1f}'.format(size), 
                        ha='center', va='bottom', fontsize=8)
    axes[0, 1].set_ylabel('크기 (MB)')
    axes[0, 1].set_title('② 모델 크기')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # 3. 메모리 분리
    x_pos = np.arange(len(precisions))
    width = 0.25
    total_memory_sum = [m + i for m, i in zip(model_memory, inference_memory)]
    bar1=axes[0, 2].bar(x_pos - width, total_memory_sum, width, label='총합', alpha=0.8, color='green')
    bar2=axes[0, 2].bar(x_pos, model_memory, width, label='모델', alpha=0.8, color='steelblue')
    bar3=axes[0, 2].bar(x_pos + width, inference_memory, width, label='추론', alpha=0.8, color='coral')
    for rect, total, model_mem, inf_mem in zip(bar1, total_memory_sum, model_memory, inference_memory):
        height = rect.get_height()
        axes[0, 2].text(rect.get_x() + rect.get_width() / 2.0, height, '{:.1f}'.format(total), 
                        ha='center', va='bottom', fontsize=8)
        axes[0, 2].text(rect.get_x() - width + rect.get_width() / 2.0, model_mem, '{:.1f}'.format(model_mem), 
                        ha='center', va='bottom', fontsize=8)
        axes[0, 2].text(rect.get_x() + width + rect.get_width() / 2.0, inf_mem, '{:.1f}'.format(inf_mem), 
                        ha='center', va='bottom', fontsize=8)
    axes[0, 2].set_ylabel('메모리 (MB)')
    axes[0, 2].set_title('③ 메모리 분리')
    axes[0, 2].set_xticks(x_pos)
    axes[0, 2].set_xticklabels(precisions, rotation=45, ha='right', fontsize=8)
    axes[0, 2].legend(fontsize=7)
    axes[0, 2].grid(axis='y', alpha=0.3)
    
    # 4. 파라미터 수
    bar=axes[1, 0].bar(precisions, total_params, alpha=0.8, color=colors)
    for rect, param in zip(bar, total_params):
        height = rect.get_height()
        axes[1, 0].text(rect.get_x() + rect.get_width() / 2.0, height, '{:.1f}'.format(param), 
                        ha='center', va='bottom', fontsize=8)
    axes[1, 0].set_ylabel('파라미터 (M)')
    axes[1, 0].set_title('④ 총 파라미터 수')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 5. FLOPs 감소율
    bar=axes[1, 1].bar(precisions, flops_reduction, alpha=0.8, color=colors)
    for rect, reduction in zip(bar, flops_reduction):
        height = rect.get_height()
        axes[1, 1].text(rect.get_x() + rect.get_width() / 2.0, height, '{:.1f}'.format(reduction), 
                        ha='center', va='bottom', fontsize=8)
    axes[1, 1].set_ylabel('감소율 (%)')
    axes[1, 1].set_title('⑤ FLOPs 감소율')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    # 6. METEOR 점수
    bar=axes[1, 2].bar(precisions, meteor_scores, alpha=0.8, color=colors)
    for rect, score in zip(bar, meteor_scores):
        height = rect.get_height()
        axes[1, 2].text(rect.get_x() + rect.get_width() / 2.0, height, '{:.2f}'.format(score), 
                        ha='center', va='bottom', fontsize=8)
    axes[1, 2].set_ylabel('METEOR')
    axes[1, 2].set_title('⑥ 캡션 품질 (METEOR)')
    axes[1, 2].tick_params(axis='x', rotation=45)
    axes[1, 2].grid(axis='y', alpha=0.3)
    
    # 7. 크기 감소율
    bar=axes[2, 0].bar(precisions, size_reduction, alpha=0.8, color=colors)
    for rect, reduction in zip(bar, size_reduction):
        height = rect.get_height()
        axes[2, 0].text(rect.get_x() + rect.get_width() / 2.0, height, '{:.1f}'.format(reduction), 
                        ha='center', va='bottom', fontsize=8)
    axes[2, 0].set_ylabel('감소율 (%)')
    axes[2, 0].set_title('⑦ 모델 크기 감소율')
    axes[2, 0].tick_params(axis='x', rotation=45)
    axes[2, 0].grid(axis='y', alpha=0.3)
    
    # 8. 메모리-성능 트레이드오프
    total_memory = [m + i for m, i in zip(model_memory, inference_memory)]
    tradeoff = [mt * mm for mt, mm in zip(ms_per_token, total_memory)]
    bar=axes[2, 1].bar(precisions, tradeoff, alpha=0.8, color=colors)
    for rect, tm in zip(bar, total_memory):
        height = rect.get_height()
        axes[2, 1].text(rect.get_x() + rect.get_width() / 2.0, height, '{:.1f}'.format(tm), 
                        ha='center', va='bottom', fontsize=8)
    axes[2, 1].set_ylabel('트레이드오프 (ms*MB)')
    axes[2, 1].set_title('⑧ 메모리-성능 트레이드오프')
    axes[2, 1].tick_params(axis='x', rotation=45)
    axes[2, 1].grid(axis='y', alpha=0.3)
    
    # 9. 전체 문장 추론 시간
    bar=axes[2, 2].bar(precisions, mean_times, alpha=0.8, color=colors)
    for rect, time in zip(bar, mean_times):
        height = rect.get_height()
        axes[2, 2].text(rect.get_x() + rect.get_width() / 2.0, height, '{:.1f}'.format(time), 
                        ha='center', va='bottom', fontsize=8)
    axes[2, 2].set_ylabel('시간 (ms)')
    axes[2, 2].set_title('⑨ 전체 문장 추론 시간')
    axes[2, 2].tick_params(axis='x', rotation=45)
    axes[2, 2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    file_name = 'embedded_metrics_finetuned.png' if ENABLE_FINETUNING else 'embedded_metrics_comprehensive.png'
    plt.savefig(os.path.join(OUTPUT_DIR, file_name), dpi=300, bbox_inches='tight')
    print("✅ Plot 저장: {}".format(os.path.join(OUTPUT_DIR, file_name)))
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
    base_model, wm, rwm = load_base_model(device=device)
    img_tensor, ref_caption = load_test_data(device=device, transform=transform)
    
    # ✅ val_dataloader 생성 (METEOR 측정용, 데이터 오염 방지)
    train_dataloader, val_dataloader = getDataset(wm)
    results = []
    
    # 2. 원본 모델 벤치마크 (Baseline)
    print("\n" + "="*70)
    print("=== [Baseline] 원본 모델 ===")
    print("="*70)
    result_baseline = run_benchmark(
        base_model, img_tensor, wm, rwm, "Original",
        ref_caption=ref_caption,
        num_runs=NUM_RUNS,
        num_meteor_images=METEO_IMAGE_NUM,
        val_dataloader=val_dataloader,
        transform=transform,
        calculate_meteor_fn=calculate_meteor
    )
    baseline_params = None
    if result_baseline:
        baseline_params = result_baseline['total_params']
        results.append(result_baseline)
    clear_memory(device)
    
    if ENABLE_FINETUNING:
        if ENABLE_QUANTIZATION:
            quantize_benchmark(base_model, img_tensor, wm, rwm, ref_caption, baseline_params, results, "Original", val_dataloader=val_dataloader)

        print("\n" + "="*70)
        print("=== Magnitude-10% & Structured-30% Pruning ===")
        pruned_model = apply_structured_pruning(
            base_model, 0.3, 
            img_tensor=img_tensor,
            device=device, use_hessian=True
        )
        pruned_model = apply_magnitude_pruning(pruned_model, 0.1)
        
        clear_memory(device)
        run_pruning_benchmark(pruned_model, "Pruning", img_tensor, wm, rwm, ref_caption, baseline_params, device, results, val_dataloader=val_dataloader)
    else:
        # 다양한 Pruning Rate로 테스트
        for pruning_rate in PRUNING_RATES:
            if ENABLE_MAGNITUDE_PRUNING:
                print("\n" + "="*70)
                print("=== Magnitude Pruning ({:.0f}%) ===".format(pruning_rate*100))
                print("="*70)
                try:
                    clear_memory(device)
                    pruned_model = apply_magnitude_pruning(base_model, pruning_rate)
                    run_pruning_benchmark(pruned_model, "Magnitude-{:.0f}%".format(pruning_rate*100), img_tensor, wm, rwm, ref_caption, baseline_params, device, results, val_dataloader=val_dataloader)
                    del pruned_model
                    clear_memory(device)
                except Exception as e:
                    print("⚠️ Magnitude Pruning ({:.0f}%) 실패: {}".format(pruning_rate*100, e))
            
            print("\n" + "="*70)
            print("=== Structured Pruning ({:.0f}%) ===".format(pruning_rate*100))
            print("="*70)
            
            if pruning_rate > MAX_PRUNING_RATE:
                print("   ⚠️ 경고: {:.0f}% 프루닝은 정확도 손실이 매우 큼".format(pruning_rate*100))
            
            try:
                clear_memory(device)
                pruned_model = apply_structured_pruning(
                    base_model, pruning_rate, 
                    img_tensor=img_tensor,
                    device=device, use_hessian=True
                )
                run_pruning_benchmark(pruned_model, "Structured-{:.0f}%".format(pruning_rate*100), img_tensor, wm, rwm, ref_caption, baseline_params, device, results, val_dataloader=val_dataloader)
                del pruned_model
                clear_memory(device)
            except Exception as e:
                print("⚠️ Structured Pruning ({:.0f}%) 실패: {}".format(pruning_rate*100, e))
    
    # 결과 요약
    print("\n" + "="*70)
    print("=== 벤치마크 결과 요약 ===")
    print("="*70)
    
    print("{'Method':<30} {'시간(ms)':<12} {'모델(MB)':<10} {'감소(%)':<10} {'METEOR':<8}")
    print("-"*80)
    for result in results:
        meteor_str = "{result.get('meteor_score', 0):.4f}" if result.get('meteor_score') else "N/A"
        print("{result['precision']:<30} "
              "{result['mean_time_ms']:.1f}±{result['std_time_ms']:.1f}  "
              "{result['model_size_mb']:.1f}       "
              "{result.get('size_reduction', 0):.1f}%      "
              "{}".format(meteor_str))
    
    # 결과 저장
    results_dict = {
        'baseline_params': baseline_params,
        'results': [{
            'precision': r['precision'],
            'mean_time_ms': float(r['mean_time_ms']),
            'std_time_ms': float(r['std_time_ms']),
            'model_size_mb': float(r['model_size_mb']),
            'meteor_score': float(r.get('meteor_score', 0)) if r.get('meteor_score') else None,
            'total_params': int(r['total_params']),
            'size_reduction': float(r.get('size_reduction', 0)),
        } for r in results]
    }
    
    results_json_path = os.path.join(OUTPUT_DIR, 'pruning_results.json')
    with open(results_json_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    print("\n✅ 결과 JSON 저장: {}".format(results_json_path))
    
    # 시각화
    plot_embedded_metrics(results)
    
    print("\n" + "="*70)
    print("=== 벤치마크 완료 ===")
    print("결과 저장 위치: {}".format(OUTPUT_DIR))
    print("="*70)


if __name__ == "__main__":
    main()
