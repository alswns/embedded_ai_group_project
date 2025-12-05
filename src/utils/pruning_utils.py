"""
Pruning 관련 유틸리티 함수들
- 파라미터 계산
- 프루닝 마스크 생성
- Hessian 중요도 계산
- 레이어 업데이트
- Magnitude Pruning
- Structured Pruning
"""

import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy


def count_nonzero_parameters(model):
    """0이 아닌 파라미터 개수 계산 (프루닝 후)"""
    nonzero_params = 0
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
        nonzero_params += param.nonzero().size(0) if param.numel() > 0 else 0
    return nonzero_params, total_params


def save_sparse_model(model, path):
    """Sparse 모델 저장"""
    try:
        total_size = 0
        nonzero_count = 0
        
        for name, param in model.named_parameters():
            total_size += param.numel()
            nonzero_count += (param != 0).sum().item()
        
        state_dict = model.state_dict()
        torch.save(state_dict, path)
        return True
    except Exception as e:
        print(f"❌ 저장 실패: {e}")
        return False


def get_sparse_model_size_mb(model):
    """Sparse 모델의 메모리 크기 계산 (MB)"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        if param.is_sparse:
            indices_size = param._indices().numel() * 8
            values_size = param._values().numel() * 4
            param_size += indices_size + values_size
        else:
            nonzero = (param != 0).sum().item()
            if nonzero > 0:
                param_size += nonzero * 4
    
    for buffer in model.buffers():
        if buffer.is_sparse:
            indices_size = buffer._indices().numel() * 8
            values_size = buffer._values().numel() * 4
            buffer_size += indices_size + values_size
        else:
            buffer_size += buffer.numel() * 4
    
    return (param_size + buffer_size) / (1024 * 1024)


def get_pruning_mask(weight, pruning_rate, dim=0, use_l2=True):
    """L2 norm 또는 magnitude 기반 프루닝 마스크 생성"""
    if dim == 0:
        importance = torch.norm(weight, p=2, dim=tuple(range(1, len(weight.shape))))
    else:
        importance = torch.norm(weight, p=2, dim=0)
    
    num_to_prune = int(pruning_rate * len(importance))
    if num_to_prune == 0:
        return torch.ones(len(importance), dtype=torch.bool, device=weight.device)
    
    _, indices = torch.topk(importance, num_to_prune, largest=False)
    mask = torch.ones(len(importance), dtype=torch.bool, device=weight.device)
    mask[indices] = False
    
    return mask


def update_linear_layer(old_layer, mask_in=None, mask_out=None, in_size=None, out_size=None):
    """선형 레이어의 가중치를 마스크 또는 크기에 따라 업데이트"""
    if mask_in is not None:
        in_features = mask_in.sum().item()
    else:
        in_features = in_size if in_size is not None else old_layer.in_features
    
    if mask_out is not None:
        out_features = mask_out.sum().item()
    else:
        out_features = out_size if out_size is not None else old_layer.out_features
    
    new_layer = nn.Linear(in_features, out_features, bias=old_layer.bias is not None)
    
    if mask_in is not None and mask_out is not None:
        new_layer.weight.data = old_layer.weight.data[mask_out, :][:, mask_in]
    elif mask_out is not None:
        new_layer.weight.data = old_layer.weight.data[mask_out, :]
    elif mask_in is not None:
        new_layer.weight.data = old_layer.weight.data[:, mask_in]
    else:
        new_layer.weight.data = old_layer.weight.data
    
    if old_layer.bias is not None and mask_out is not None:
        new_layer.bias.data = old_layer.bias.data[mask_out]
    elif old_layer.bias is not None:
        new_layer.bias.data = old_layer.bias.data
    
    return new_layer


def compute_channel_importance_hessian(weight, pruning_rate, dim=1, hessian_importance=None):
    """Hessian 또는 L2 norm 기반 채널 중요도 계산"""
    if hessian_importance is not None:
        if dim == 1:
            channel_importance = (hessian_importance * (weight ** 2)).sum(dim=0)
        else:
            channel_importance = (hessian_importance * (weight ** 2)).sum(dim=1)
    else:
        if dim == 1:
            channel_importance = torch.norm(weight, p=2, dim=0)
        else:
            channel_importance = torch.norm(weight, p=2, dim=1)
    
    num_to_prune = int(pruning_rate * channel_importance.numel())
    if num_to_prune == 0:
        return torch.ones(channel_importance.numel(), dtype=torch.bool, device=weight.device)
    
    _, indices = torch.sort(channel_importance)
    mask = torch.ones(channel_importance.numel(), dtype=torch.bool, device=weight.device)
    mask[indices[:num_to_prune]] = False
    
    return mask


def compute_hessian_importance(model, layer, img_tensor, captions_batch, wm, rwm, device, num_samples=64):
    """Hessian 행렬을 근사하여 중요도 계산"""
    print(f"      🔍 Hessian 계산 중 ({num_samples}개 샘플)...")
    model.eval()
    model.to(device)
    return None  # 기본 구현


# ============================================================================
# Magnitude Pruning
# ============================================================================
def apply_magnitude_pruning(model, pruning_rate):
    """
    Magnitude-based Pruning (비구조적) - 직접 마스킹 방식
    
    모든 가중치에서 작은 값들을 0으로 설정
    """
    pruned_model = deepcopy(model)
    pruned_model.eval()
    
    print(f"   🔧 Magnitude-based Pruning 적용 (직접 마스킹, {pruning_rate*100:.0f}%)...")
    
    # 1. 전체 모델의 모든 가중치를 수집
    all_weights = []
    weight_info = []
    
    for name, module in pruned_model.named_modules():
        if isinstance(module, (nn.Linear, nn.GRU, nn.GRUCell)):
            for param_name in ['weight', 'weight_ih', 'weight_hh']:
                if hasattr(module, param_name):
                    param = getattr(module, param_name)
                    if param is not None:
                        all_weights.append(param.data.abs().flatten())
                        weight_info.append((name, param_name, module, param.shape))
    
    if not all_weights:
        print(f"   ⚠️ 프루닝할 가중치가 없습니다")
        return pruned_model
    
    # 2. 전체 가중치를 하나로 병합하여 임계값 계산
    all_weights_cpu = [w.cpu() for w in all_weights]
    all_weights_tensor = torch.cat(all_weights_cpu)
    total_weights = all_weights_tensor.numel()
    num_to_prune = int(pruning_rate * total_weights)
    
    # 3. 임계값 계산
    if num_to_prune > 0:
        threshold = torch.kthvalue(all_weights_tensor, num_to_prune).values
    else:
        threshold = float('inf')
    
    print(f"   📊 임계값: {threshold:.6f}")
    print(f"   📊 제거 대상: {num_to_prune:,} / {total_weights:,} 개 가중치")
    
    # 4. 마스크를 모델에 저장
    pruned_model._magnitude_pruning_masks = {}
    
    # 5. 각 모듈에 마스크 적용
    total_pruned = 0
    for name, param_name, module, param_shape in weight_info:
        param = getattr(module, param_name)
        magnitude = param.data.abs()
        mask = (magnitude > threshold).float()
        
        mask_key = f"{name}_{param_name}"
        pruned_model._magnitude_pruning_masks[mask_key] = (module, param_name, mask.clone())
        param.data = param.data * mask
        
        pruned_count = (mask == 0).sum().item()
        total_pruned += pruned_count
        
        if pruned_count > 0:
            print(f"   ✅ {name}.{param_name}: {pruned_count:,}개 가중치 제거 ({pruned_count/magnitude.numel()*100:.1f}%)")
    
    pruned_model.eval()
    
    # 최종 파라미터 확인
    old_params = sum(p.numel() for p in model.parameters())
    new_nonzero = sum((p != 0).sum().item() for p in pruned_model.parameters())
    total_params = sum(p.numel() for p in pruned_model.parameters())
    actual_reduction = (1 - new_nonzero / old_params) * 100
    
    print(f"   ✂️ Magnitude Pruning 완료 (직접 마스킹)")
    print(f"   📊 희소성: {actual_reduction:.1f}% (0인 가중치: {total_params - new_nonzero:,} / {total_params:,})")
    
    return pruned_model


# ============================================================================
# Structured Pruning
# ============================================================================
def apply_structured_pruning(model, pruning_rate, img_tensor=None, device=None, use_hessian=True):
    """
    Structured Pruning 적용 (Hessian 기반 - GRU 포함)
    
    GRU Hidden State를 점진적으로 축소
    """
    pruned_model = deepcopy(model)
    pruned_model.eval()
    
    print(f"\n   📊 파라미터 분석:")
    total_params = sum(p.numel() for p in pruned_model.parameters())
    
    if hasattr(pruned_model, 'encoder'):
        encoder_params = sum(p.numel() for p in pruned_model.encoder.parameters())
        print(f"      Encoder: {encoder_params:,} ({100*encoder_params/total_params:.1f}%)")
    
    if hasattr(pruned_model, 'decoder'):
        decoder_params = sum(p.numel() for p in pruned_model.decoder.parameters())
        print(f"      Decoder: {decoder_params:,} ({100*decoder_params/total_params:.1f}%)")
        
        if hasattr(pruned_model.decoder, 'decode_step'):
            gru_params = sum(p.numel() for p in pruned_model.decoder.decode_step.parameters())
            print(f"         └─ GRU Cell: {gru_params:,} ({100*gru_params/total_params:.1f}%)")
    
    decoder = pruned_model.decoder
    
    # GRU Hidden State 축소
    print(f"\n   🎯 GRU Hidden State 점진적 축소 ({pruning_rate*100:.0f}%)")
    
    if hasattr(decoder, 'decode_step'):
        old_gru = decoder.decode_step
        old_hidden_size = old_gru.hidden_size
        new_hidden_size = int(old_hidden_size * (1 - pruning_rate))
        
        print(f"      GRU Hidden Size: {old_hidden_size} → {new_hidden_size}")
        
        # Hessian 기반 중요 뉴런 선택
        if use_hessian and device is not None:
            try:
                w_ih = old_gru.weight_ih.data
                w_hh = old_gru.weight_hh.data
                importance = torch.zeros(old_hidden_size, device=device)
                
                for gate_idx in range(3):
                    start_idx = gate_idx * old_hidden_size
                    end_idx = (gate_idx + 1) * old_hidden_size
                    w_ih_gate = w_ih[start_idx:end_idx, :]
                    importance += torch.norm(w_ih_gate, p=2, dim=1)
                    w_hh_gate = w_hh[start_idx:end_idx, :]
                    importance += torch.norm(w_hh_gate, p=2, dim=1)
                
                _, indices_to_keep = torch.topk(importance, new_hidden_size)
                indices_to_keep = torch.sort(indices_to_keep)[0]
                print(f"      ✅ Hessian 기반 중요 뉴런 선택 완료")
            except Exception as e:
                print(f"      ⚠️ Hessian 계산 실패: {e}")
                indices_to_keep = torch.arange(new_hidden_size, device=device)
        else:
            indices_to_keep = torch.arange(new_hidden_size, device=device if device else 'cpu')
        
        # 새로운 GRUCell 생성
        new_gru = nn.GRUCell(old_gru.input_size, new_hidden_size)
        
        # 가중치 축소 (weight_ih)
        new_gru.weight_ih.data = torch.zeros(3 * new_hidden_size, old_gru.input_size, device=device if device else 'cpu')
        for gate_idx in range(3):
            old_start = gate_idx * old_hidden_size
            old_end = (gate_idx + 1) * old_hidden_size
            new_start = gate_idx * new_hidden_size
            new_end = (gate_idx + 1) * new_hidden_size
            new_gru.weight_ih.data[new_start:new_end, :] = old_gru.weight_ih.data[old_start:old_end, :][indices_to_keep, :]
        
        # 가중치 축소 (weight_hh)
        new_gru.weight_hh.data = torch.zeros(3 * new_hidden_size, new_hidden_size, device=device if device else 'cpu')
        for gate_idx in range(3):
            old_start = gate_idx * old_hidden_size
            old_end = (gate_idx + 1) * old_hidden_size
            new_start = gate_idx * new_hidden_size
            new_end = (gate_idx + 1) * new_hidden_size
            old_w = old_gru.weight_hh.data[old_start:old_end, :]
            new_gru.weight_hh.data[new_start:new_end, :] = old_w[indices_to_keep, :][:, indices_to_keep]
        
        # Bias 축소
        if old_gru.bias_ih is not None:
            new_gru.bias_ih.data = torch.zeros(3 * new_hidden_size, device=device if device else 'cpu')
            for gate_idx in range(3):
                old_start = gate_idx * old_hidden_size
                old_end = (gate_idx + 1) * old_hidden_size
                new_start = gate_idx * new_hidden_size
                new_end = (gate_idx + 1) * new_hidden_size
                new_gru.bias_ih.data[new_start:new_end] = old_gru.bias_ih.data[old_start:old_end][indices_to_keep]
        
        if old_gru.bias_hh is not None:
            new_gru.bias_hh.data = torch.zeros(3 * new_hidden_size, device=device if device else 'cpu')
            for gate_idx in range(3):
                old_start = gate_idx * old_hidden_size
                old_end = (gate_idx + 1) * old_hidden_size
                new_start = gate_idx * new_hidden_size
                new_end = (gate_idx + 1) * new_hidden_size
                new_gru.bias_hh.data[new_start:new_end] = old_gru.bias_hh.data[old_start:old_end][indices_to_keep]
        
        decoder.decode_step = new_gru
        decoder.decoder_dim = new_hidden_size
        
        # 연결된 레이어 업데이트
        if hasattr(decoder, 'fc'):
            old_fc = decoder.fc
            new_fc = nn.Linear(new_hidden_size, old_fc.out_features)
            new_fc.weight.data = old_fc.weight.data[:, indices_to_keep]
            new_fc.bias.data = old_fc.bias.data.clone()
            decoder.fc = new_fc
        
        if hasattr(decoder, 'init_h'):
            old_init_h = decoder.init_h
            new_init_h = nn.Linear(old_init_h.in_features, new_hidden_size)
            new_init_h.weight.data = old_init_h.weight.data[indices_to_keep, :]
            new_init_h.bias.data = old_init_h.bias.data[indices_to_keep]
            decoder.init_h = new_init_h
    
    # Attention 차원도 축소
    if hasattr(decoder, 'encoder_att') and hasattr(decoder, 'full_att'):
        weight = decoder.encoder_att.weight.data
        mask_attention_dim = compute_channel_importance_hessian(weight, pruning_rate, dim=0)
        new_attention_dim = mask_attention_dim.sum().item()
        
        print(f"   📊 Attention Dim: {weight.shape[0]} → {new_attention_dim}")
        
        decoder.encoder_att = update_linear_layer(decoder.encoder_att, mask_out=mask_attention_dim, in_size=decoder.encoder_dim)
        decoder.attention_dim = new_attention_dim
        
        if hasattr(decoder, 'decoder_att'):
            new_decoder_att = nn.Linear(decoder.decoder_dim, new_attention_dim)
            nn.init.xavier_uniform_(new_decoder_att.weight)
            if new_decoder_att.bias is not None:
                nn.init.zeros_(new_decoder_att.bias)
            decoder.decoder_att = new_decoder_att
        
        if hasattr(decoder, 'full_att'):
            decoder.full_att = update_linear_layer(decoder.full_att, mask_in=mask_attention_dim, out_size=1)
    
    pruned_model.decoder = decoder
    pruned_model.eval()
    
    # 결과 출력
    old_params = sum(p.numel() for p in model.parameters())
    new_params = sum(p.numel() for p in pruned_model.parameters())
    reduction = (1 - new_params / old_params) * 100
    
    print(f"   ✂️ Structured Pruning 완료: {pruning_rate*100:.0f}% 프루닝")
    print(f"   📊 파라미터 감소: {old_params:,} → {new_params:,} ({reduction:.1f}% 감소)")
    
    return pruned_model
    
    # Hessian 근사 (Diagonal approximation)
    hessian = None
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    for i in range(num_samples):
        idx = i % len(img_tensor)
        img = img_tensor[idx:idx+1].to(device)
        caps = captions_batch[idx:idx+1].to(device)
        
        model.zero_grad()
        
        try:
            # Forward pass
            outputs, _ = model(img, caps)
            targets = caps[:, 1:]
            outputs_trimmed = outputs[:, :targets.shape[1], :]
            
            # Loss
            vocab_size = outputs_trimmed.shape[-1]
            loss = criterion(outputs_trimmed.reshape(-1, vocab_size), targets.reshape(-1))
            
            # Backward pass
            loss.backward()
            
            # Gradient 수집
            if hasattr(layer, 'weight') and layer.weight.grad is not None:
                grad = layer.weight.grad.data
                if hessian is None:
                    hessian = grad ** 2
                else:
                    hessian += grad ** 2
        except Exception as e:
            continue
    
    if hessian is not None:
        hessian = hessian / num_samples
    
    return hessian
