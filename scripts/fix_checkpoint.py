"""
프루닝된 모델 체크포인트 복구 스크립트
원본 모델과 프루닝된 모델의 호환성을 맞추기 위해 체크포인트를 수정합니다.
"""
import torch
import os

def fix_pruned_checkpoint():
    """프루닝된 체크포인트 복구"""
    checkpoint_path = "pruning_results/Pruning_epoch_1_checkpoint.pt"
    
    if not os.path.exists(checkpoint_path):
        print("❌ 체크포인트를 찾을 수 없습니다: {}".format(checkpoint_path))
        return
    
    print("📂 체크포인트 로드 중: {}".format(checkpoint_path))
    
    # Python/PyTorch 버전 호환성
    try:
        # Python 3.11+: weights_only 파라미터 필요
        checkpoint = torch.load(checkpoint_path, weights_only=False)
    except TypeError:
        # Python 3.6-3.10: weights_only 파라미터 미지원
        checkpoint = torch.load(checkpoint_path)
    
    # 체크포인트 구조 확인
    print("\n📋 체크포인트 구조:")
    for key in checkpoint.keys():
        print("   • {}".format(key))
    
    # model_state_dict 크기 정보 출력
    if 'model_state_dict' in checkpoint:
        print("\n📊 모델 레이어 정보:")
        state_dict = checkpoint['model_state_dict']
        
        # 중요 레이어 크기 확인
        decoder_keys = [k for k in state_dict.keys() if 'decoder' in k]
        print("\n   Decoder 레이어 ({}개):".format(len(decoder_keys)))
        for key in sorted(decoder_keys)[:5]:
            print("      • {}: {}".format(key, state_dict[key].shape))
        
        # 원본 모델과의 차이점 파악
        print("\n💡 체크포인트 수정:")
        
        # 음성 모델과 호환되도록 메타데이터 추가
        if 'decoder_dim' not in checkpoint:
            # state_dict에서 decoder_dim 추출
            if 'decoder.decode_step.weight_ih' in state_dict:
                decoder_dim = state_dict['decoder.decode_step.weight_ih'].shape[0] // 3
                checkpoint['decoder_dim'] = decoder_dim
                print("   ✓ decoder_dim 추가: {}".format(decoder_dim))
        
        if 'attention_dim' not in checkpoint:
            if 'decoder.encoder_att.weight' in state_dict:
                attention_dim = state_dict['decoder.encoder_att.weight'].shape[0]
                checkpoint['attention_dim'] = attention_dim
                print("   ✓ attention_dim 추가: {}".format(attention_dim))
        
        # 저장
        torch.save(checkpoint, checkpoint_path)
        print("\n✅ 체크포인트 수정 완료: {}".format(checkpoint_path))
    else:
        print("❌ model_state_dict를 찾을 수 없습니다.")

if __name__ == "__main__":
    fix_pruned_checkpoint()
