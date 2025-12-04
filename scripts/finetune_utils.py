"""
파인튜닝 관련 헬퍼 함수들
"""

import os
import torch


def load_checkpoint(label, device):
    """체크포인트 로드 및 시작 epoch 반환"""
    checkpoint_files = [f for f in os.listdir("pruning_results") 
                        if f.startswith(label) and f.endswith('_checkpoint.pt')]
    
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
    checkpoint_path = os.path.join("pruning_results", latest_file)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        return checkpoint, latest_epoch, checkpoint_path
    except Exception as e:
        print(f"   ⚠️ 체크포인트 로드 실패: {e}")
        return None, 0, None


def setup_training(model, learning_rate, device):
    """학습 설정: Encoder Freeze, Optimizer 생성"""
    # Encoder Freeze
    if hasattr(model, 'encoder'):
        for param in model.encoder.parameters():
            param.requires_grad = False
        print(f"   🔒 Encoder Freeze: CNN 파라미터 학습 금지")
    
    # Optimizer 설정
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
    
    print(f"   📚 학습률: {learning_rate}")
    
    # 학습할 파라미터 개수
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    print(f"   📊 학습 대상 파라미터: {trainable_count:,} / {total_count:,} ({100*trainable_count/total_count:.1f}%)")
    
    return optimizer, criterion


def save_checkpoint(model, optimizer, epoch, label, avg_loss, avg_val_loss, meteor_score):
    """체크포인트 저장"""
    os.makedirs("pruning_results", exist_ok=True)
    checkpoint_path = os.path.join("pruning_results", f"{label}_epoch_{epoch+1}_checkpoint.pt")
    
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'avg_loss': avg_loss,
        'avg_val_loss': avg_val_loss,
        'meteor_score': meteor_score,
    }
    torch.save(checkpoint, checkpoint_path)
    
    print(f"      💾 체크포인트 저장 완료")
    return checkpoint_path


def print_checkpoint_info(checkpoint, latest_epoch):
    """체크포인트 정보 출력"""
    print(f"   📂 체크포인트 발견: Epoch {latest_epoch}")
    if checkpoint.get('avg_loss'):
        print(f"   📊 이전 평균 Loss: {checkpoint['avg_loss']:.4f}")


def restore_optimizer(optimizer, optimizer_state):
    """Optimizer State 복구"""
    if optimizer_state is None:
        return
    
    try:
        optimizer.load_state_dict(optimizer_state)
        print(f"   ✅ Optimizer State 복구 완료")
    except Exception as e:
        print(f"   ⚠️ Optimizer State 복구 실패: {e}")
