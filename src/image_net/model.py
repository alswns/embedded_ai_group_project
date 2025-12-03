import torch
import torch.nn as nn
from torchvision import models

class MobileNetV3Encoder(nn.Module):
    def __init__(self, model_type='small', pretrained=True):
        super(MobileNetV3Encoder, self).__init__()
        
        # 1. 모델 로드 (Pretrained Weights 사용)
        if model_type == 'large':
            # 좀 더 정확함 (하지만 무거움)
            # 출력 채널: 960
            base_model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
            self.out_channels = 960 
        else:
            # 가장 빠름 (임베디드 강력 추천)
            # 출력 채널: 576
            base_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
            self.out_channels = 576

        # 2. 피처 추출 부분만 가져오기
        # 마지막 분류기(classifier)와 풀링(pooling) 층을 제외한 앞부분('features')만 사용
        self.features = base_model.features
        
        # 학습된 가중치를 그대로 쓸 것이므로 얼림 (Freeze)
        # 데이터가 충분하다면 True로 바꿔서 미세조정 가능
        for param in self.features.parameters():
            param.requires_grad = False 

    def forward(self, images):
        # 입력: [Batch, 3, 224, 224] (일반적인 이미지 크기)
        out = self.features(images)
        
        # 출력: [Batch, 576, 7, 7] (Small 기준)
        return out

# --- 사용 예시 ---
if __name__ == "__main__":
    # 1. 인코더 생성 (Small 버전)
    encoder = MobileNetV3Encoder(model_type='small')
    
    # 2. 가짜 이미지 입력
    dummy_img = torch.randn(1, 3, 224, 224)
    
    # 3. 피처맵 추출
    features = encoder(dummy_img)
    
    print(f"MobileNetV3 출력 크기: {features.shape}")
    # 결과: torch.Size([1, 576, 7, 7]) -> 채널이 576개, 크기는 7x7