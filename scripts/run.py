import torch
from src.muti_modal_model.model import MobileNetCaptioningModel

if __name__ == "__main__":
    # 임베디드 장치 설정 (없으면 cpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 단어장 가상 생성
    vocab = {'<pad>':0, '<start>':1, '<end>':2, 'a':3, 'cat':4}
    rev_vocab = {v:k for k,v in vocab.items()}
    
    # 1. 모델 생성 (width_mult=0.5로 더 경량화했다고 가정)
    model = MobileNetCaptioningModel(vocab_size=len(vocab)).to(device)
    
    # 2. 가짜 데이터 준비
    img = torch.randn(2, 3, 224, 224).to(device) # 이미지 2장
    caps = torch.LongTensor([[1, 3, 4, 2], [1, 3, 4, 2]]).to(device) # 캡션 2개
    
    # 3. 학습 시 (Forward)
    outputs, _ = model(img, caps)
    print("학습 출력 크기:", outputs.shape) # [2, 3, 5]
    
    # 4. 추론 시 (Generate)
    one_img = torch.randn(1, 3, 224, 224).to(device)
    sentence = model.generate(one_img, vocab, rev_vocab)
    print("생성된 문장:", sentence)