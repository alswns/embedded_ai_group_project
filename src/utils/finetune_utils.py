"""
파인튜닝 관련 헬퍼 함수들
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def load_checkpoint(label, device, checkpoint_dir="pruning_results"):
    """
    체크포인트 로드 및 시작 epoch 반환
    
    Returns:
        tuple: (checkpoint_dict, start_epoch, checkpoint_path)
    """
    if not os.path.exists(checkpoint_dir):
        return None, 0, None
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                        if f.startswith(label) and f.endswith('')]
    
    if not checkpoint_files:
        return None, 0, None
    
    # 가장 최신 체크포인트 찾기
    epoch_numbers = []
    for f in checkpoint_files:
        
        try:
            epoch_num = int(f.split('epoch_')[-1].replace('_checkpoint.pt', ''))
            epoch_numbers.append((epoch_num, f))
        except ValueError:
            continue
    
    if not epoch_numbers:
        return None, 0, None
    
    epoch_numbers.sort()
    latest_epoch, latest_file = epoch_numbers[-1]
    checkpoint_path = os.path.join(checkpoint_dir, latest_file)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        start_epoch = checkpoint.get('epoch', latest_epoch)
        return checkpoint, start_epoch, checkpoint_path
    except Exception as e:
        print(f"   ⚠️ 체크포인트 로드 실패: {e}")
        return None, 0, None


def setup_training(model, learning_rate, device, freeze_encoder=True):
    """학습 설정: Encoder Freeze, Optimizer 생성"""
    # Encoder Freeze
    if freeze_encoder and hasattr(model, 'encoder'):
        for param in model.encoder.parameters():
            param.requires_grad = False
        print(f"   🔒 Encoder Freeze: CNN 파라미터 학습 금지")
    
    # Optimizer 설정
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
    
    print(f"   📚 학습률: {learning_rate}")
    
    # 학습할 파라미터 개수
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    print(f"   📊 학습 대상 파라미터: {trainable_count:,} / {total_count:,} ({100*trainable_count/total_count:.1f}%)")
    
    return optimizer, criterion


def save_checkpoint(model, optimizer, epoch, label, word_map, rev_word_map, vocab_size,
                   avg_train_loss=None, avg_val_loss=None, meteor_score=None,
                   checkpoint_dir="pruning_results"):
    """체크포인트 저장 (통일된 형식)"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{label}_epoch_{epoch+1}_checkpoint.pt")
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'word_map': word_map,
        'rev_word_map': rev_word_map,
        'vocab_size': vocab_size,
        'epoch': epoch + 1,
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    if avg_train_loss is not None:
        checkpoint['train_loss'] = avg_train_loss
    if avg_val_loss is not None:
        checkpoint['val_loss'] = avg_val_loss
    if meteor_score is not None:
        checkpoint['meteor_score'] = meteor_score
    
    torch.save(checkpoint, checkpoint_path)
    print(f"      💾 체크포인트 저장 완료: {checkpoint_path}")
    return checkpoint_path


def print_checkpoint_info(checkpoint, latest_epoch):
    """체크포인트 정보 출력"""
    print(f"   📂 체크포인트 발견: Epoch {latest_epoch}")
    
    if 'train_loss' in checkpoint:
        print(f"   📊 이전 학습 Loss: {checkpoint['train_loss']:.4f}")
    if 'val_loss' in checkpoint:
        print(f"   📊 이전 검증 Loss: {checkpoint['val_loss']:.4f}")
    if 'meteor_score' in checkpoint:
        print(f"   ⭐ 이전 METEOR: {checkpoint['meteor_score']:.4f}")
    if 'vocab_size' in checkpoint:
        print(f"   📚 어휘집 크기: {checkpoint['vocab_size']:,}")


def restore_optimizer(optimizer, optimizer_state):
    """Optimizer State 복구"""
    if optimizer_state is None:
        return
    
    try:
        optimizer.load_state_dict(optimizer_state)
        print(f"   ✅ Optimizer State 복구 완료")
    except Exception as e:
        print(f"   ⚠️ Optimizer State 복구 실패: {e}")


def load_model_checkpoint(checkpoint_path, device):
    """저장된 모델 체크포인트 로드"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        print(f"✅ 체크포인트 로드 완료: {checkpoint_path}")
        
        if 'epoch' in checkpoint:
            print(f"   📂 Epoch: {checkpoint['epoch']}")
        if 'vocab_size' in checkpoint:
            print(f"   📚 어휘집 크기: {checkpoint['vocab_size']:,}")
        if 'train_loss' in checkpoint:
            print(f"   📊 학습 Loss: {checkpoint['train_loss']:.4f}")
        if 'val_loss' in checkpoint:
            print(f"   📊 검증 Loss: {checkpoint['val_loss']:.4f}")
        
        return checkpoint
    except Exception as e:
        print(f"❌ 체크포인트 로드 실패: {e}")
        return None


def load_model_from_checkpoint(checkpoint_path, model, device):
    """체크포인트에서 모델과 word_map 로드"""
    checkpoint = load_model_checkpoint(checkpoint_path, device)
    if checkpoint is None:
        return None, None, None, None
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"   ✅ 모델 가중치 로드 완료")
    
    word_map = checkpoint.get('word_map')
    rev_word_map = checkpoint.get('rev_word_map')
    vocab_size = checkpoint.get('vocab_size')
    
    return model, word_map, rev_word_map, vocab_size


def apply_magnitude_mask(model):
    """Magnitude Pruning의 마스크 강제 적용 (매 배치마다 호출)"""
    if hasattr(model, '_magnitude_pruning_masks'):
        for mask_key, (module, param_name, mask) in model._magnitude_pruning_masks.items():
            param = getattr(module, param_name)
            param.data = param.data * mask.to(param.device)


def fine_tune_model(model, train_dataloader, val_dataloader, word_map, device,
                   epochs=10, learning_rate=5e-5, label="finetuned",
                   early_stopping_patience=3, benchmark_fn=None,
                   img_tensor=None, wm=None, rwm=None, ref_caption=None, baseline_params=None):
    """
    파인튜닝 수행 (통합 함수)
    
    Args:
        model: 파인튜닝할 모델
        train_dataloader: 학습 데이터로더
        val_dataloader: 검증 데이터로더
        word_map: 단어 → 인덱스 매핑
        device: 디바이스
        epochs: 에포크 수
        learning_rate: 학습률
        label: 체크포인트 레이블
        early_stopping_patience: Early Stopping 인내심
        benchmark_fn: 벤치마크 함수 (optional)
        img_tensor, wm, rwm, ref_caption, baseline_params: 벤치마크용 파라미터
    
    Returns:
        model: 파인튜닝된 모델
    """
    print(f"\n   🔄 파인 튜닝 시작 ({epochs} epoch)...")
    
    # 체크포인트 로드
    checkpoint, start_epoch, checkpoint_path = load_checkpoint(label, device)
    optimizer_state = checkpoint.get('optimizer_state_dict') if checkpoint else None
    
    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print_checkpoint_info(checkpoint, start_epoch)
        print(f"   ✅ Epoch {start_epoch+1}부터 재개합니다.")
    else:
        print(f"   ℹ️ 처음부터 시작합니다.")
    
    # 학습 설정
    optimizer, criterion = setup_training(model, learning_rate, device)
    restore_optimizer(optimizer, optimizer_state)
    
    # 모델 설정
    model.train()
    model.to(device)
    
    vocab_size = len(word_map)
    rev_word_map = {v: k for k, v in word_map.items()}
    
    # Early Stopping 설정
    best_meteor_score = -float('inf')
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # 파인튜닝 진행
    for epoch in range(start_epoch, epochs):
        print(f"   🏋️ Epoch {epoch+1}/{epochs}")
        total_loss = 0
        num_batches = 0
        
        train_iter = tqdm(enumerate(train_dataloader), total=len(train_dataloader), 
                         desc=f"      학습 중", ncols=100)
        
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
                
                # Magnitude Pruning 마스크 강제 적용
                apply_magnitude_mask(model)
                
                total_loss += loss.item()
                num_batches += 1
            except Exception as e:
                print(f"   ⚠️ 배치 {batch_idx} 학습 실패: {e}")
                continue
            
            if (batch_idx + 1) % 10 == 0:
                train_iter.set_postfix(loss=f"{total_loss / num_batches:.4f}")
        
        # Epoch 완료
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        print(f"   ✅ Epoch {epoch+1} 완료 (학습 Loss: {avg_loss:.4f})")
        
        # 검증
        print(f"   📊 검증 데이터 평가 중...")
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
        print(f"      검증 Loss: {avg_val_loss:.4f}")
        
        model.train()
        
        # 벤치마크 실행 (옵션)
        current_meteor_score = None
        if benchmark_fn and img_tensor is not None and wm is not None and rwm is not None:
            print(f"\n   📊 Epoch {epoch+1} 벤치마크 시작...")
            model.eval()
            benchmark_result = benchmark_fn(
                model, img_tensor, wm, rwm,
                f"Fine-tuned (Epoch {epoch+1}/{epochs})",
                ref_caption=ref_caption,
                baseline_params=baseline_params
            )
            model.train()
            
            if benchmark_result and benchmark_result.get('meteor_score'):
                current_meteor_score = benchmark_result['meteor_score']
                print(f"      ⭐ METEOR: {current_meteor_score:.4f}")
        
        # Early Stopping 체크
        if current_meteor_score is not None and best_meteor_score is not None:
            
            if current_meteor_score > best_meteor_score:
                best_meteor_score = current_meteor_score
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                print(f"   🎉 새로운 최고 METEOR 점수: {best_meteor_score:.4f}")
            elif avg_val_loss < best_loss:
                best_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                print(f"   🎉 새로운 최저 검증 Loss: {best_loss:.4f}")
                
            else:
                patience_counter += 1
                print(f"   ⚠️ METEOR 미개선 (Patience: {patience_counter}/{early_stopping_patience})")
                print(f"   ⚠️ 검증 Loss 미개선 (Patience: {patience_counter}/{early_stopping_patience})")
                
                if patience_counter >= early_stopping_patience:
                    print(f"\n   🛑 Early Stopping 발동! Epoch {epoch+1}에서 학습 종료")
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
    
    print(f"\n   ✅ 파인 튜닝 완료")
    return model
