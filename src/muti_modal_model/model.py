import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from src.gru_model.model import LightweightCaptionDecoder

class MobileNetCaptioningModel(nn.Module):
    def __init__(self, vocab_size, width_mult=1.0, embed_dim=256, decoder_dim=512, attention_dim=256):
        super(MobileNetCaptioningModel, self).__init__()
        
        # [A] 인코더 설정 (MobileNetV3 Small)
        # width_mult를 조절해서 더 경량화 가능 (예: 0.5)
        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT, width_mult=width_mult)
        
        # 마지막 분류기(classifier)를 떼어내고, 특징 추출기(features)만 가져옴
        self.encoder = mobilenet.features
        
        # 학습된 가중치 고정 (Fine-tuning 시에는 True로 변경)
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # Encoder 출력 채널 수 계산 (Small 기준: 576 * width_mult)
        # 기본(1.0)이면 576, 0.5면 288 등
        sample_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output_channels = self.encoder(sample_input).shape[1]
            
        print("Encoder Output Channels: {}".format(output_channels))

        # [B] 디코더 설정 (Encoder 출력 채널 수를 그대로 입력으로 받음)
        self.decoder = LightweightCaptionDecoder(
            attention_dim=attention_dim,
            embed_dim=embed_dim,  # 임베딩 차원을 파라미터로 받음
            decoder_dim=decoder_dim,
            vocab_size=vocab_size,
            encoder_dim=output_channels  # <--- 여기가 핵심 연결 고리!
        )

    def forward(self, images, captions):
        """
        학습(Training)을 위한 Forward 함수
        """
        # 1. 이미지 -> 피처맵 추출
        # images: [Batch, 3, 224, 224] -> features: [Batch, 576, 7, 7]
        features = self.encoder(images)
        
        # 2. 피처맵 평탄화 (Flatten)
        # [Batch, 576, 7, 7] -> [Batch, 576, 49] -> [Batch, 49, 576]
        # Attention 모듈은 (Batch, 픽셀수, 채널) 형태를 원함
        batch_size = features.size(0)
        channel = features.size(1)
        features = features.view(batch_size, channel, -1).permute(0, 2, 1)
        
        # 3. 디코더에 넣어서 문장 생성
        outputs, alphas = self.decoder(features, captions)
        
        return outputs, alphas

    def generate(self, image, word_map, rev_word_map, max_len=20):
        """
        실제 사용(Inference)을 위한 함수 (이미지 1장 -> 문장)
        """
        self.eval()
        with torch.no_grad():
            # 1. 인코더 통과
            features = self.encoder(image) # [1, 576, 7, 7]
            
            # 2. 차원 변환
            batch_size = features.size(0)
            channel = features.size(1)
            encoder_out = features.view(batch_size, channel, -1).permute(0, 2, 1)
            
            # 3. 디코더 초기화
            start_token = word_map['<start>']
            input_word = torch.LongTensor([start_token]).to(image.device)
            h = self.decoder.init_h(encoder_out.mean(dim=1))
            
            seq = []
            
            # 4. 단어 생성 루프
            for i in range(max_len):
                embeddings = self.decoder.embedding(input_word)
                attention_weighted_encoding, _ = self.decoder.attention(encoder_out, h)
                gru_input = torch.cat([embeddings, attention_weighted_encoding], dim=1)
                h = self.decoder.decode_step(gru_input, h)
                
                preds = self.decoder.fc(h)
                predicted_id = preds.argmax(dim=1)
                
                seq.append(predicted_id.item())
                if rev_word_map[predicted_id.item()] == '<end>':
                    break
                input_word = predicted_id
                
            return [rev_word_map[k] for k in seq]