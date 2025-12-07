
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.gru_model.model import LightweightCaptionDecoder
import gc
# class MobileNetCaptioningModel(nn.Module):
#     def __init__(self, vocab_size, width_mult=1.0, embed_dim=256, decoder_dim=512, attention_dim=256):
#         super(MobileNetCaptioningModel, self).__init__()
        
#         # [A] 인코더 설정 (MobileNetV3 Small)
#         # width_mult를 조절해서 더 경량화 가능 (예: 0.5)
#         from torchvision import models
#         mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT, width_mult=width_mult)
        
#         # 마지막 분류기(classifier)를 떼어내고, 특징 추출기(features)만 가져옴
#         self.encoder = mobilenet.features
        
#         # 학습된 가중치 고정 (Fine-tuning 시에는 True로 변경)
#         for param in self.encoder.parameters():
#             param.requires_grad = False
            
#         # Encoder 출력 채널 수 계산 (Small 기준: 576 * width_mult)
#         # 기본(1.0)이면 576, 0.5면 288 등
#         sample_input = torch.randn(1, 3, 224, 224)
#         with torch.no_grad():
#             output_channels = self.encoder(sample_input).shape[1]
            
#         print("Encoder Output Channels: {}".format(output_channels))

#         # [B] 디코더 설정 (Encoder 출력 채널 수를 그대로 입력으로 받음)
#         self.decoder = LightweightCaptionDecoder(
#             attention_dim=attention_dim,
#             embed_dim=embed_dim,  # 임베딩 차원을 파라미터로 받음
#             decoder_dim=decoder_dim,
#             vocab_size=vocab_size,
#             encoder_dim=output_channels  # <--- 여기가 핵심 연결 고리!
#         )

#     def forward(self, images, captions):
#         """
#         학습(Training)을 위한 Forward 함수
#         """
#         # 1. 이미지 -> 피처맵 추출
#         # images: [Batch, 3, 224, 224] -> features: [Batch, 576, 7, 7]
#         features = self.encoder(images)
        
#         # 2. 피처맵 평탄화 (Flatten)
#         # [Batch, 576, 7, 7] -> [Batch, 576, 49] -> [Batch, 49, 576]
#         # Attention 모듈은 (Batch, 픽셀수, 채널) 형태를 원함
#         batch_size = features.size(0)
#         channel = features.size(1)
#         features = features.view(batch_size, channel, -1).permute(0, 2, 1)
        
#         # 3. 디코더에 넣어서 문장 생성
#         outputs, alphas = self.decoder(features, captions)
        
#         return outputs, alphas

#     def generate(self, image, word_map, rev_word_map, max_len=20):
#         """
#         실제 사용(Inference)을 위한 함수 (이미지 1장 -> 문장)
#         """
#         self.eval()
#         with torch.no_grad():
#             # 1. 인코더 통과
#             features = self.encoder(image) # [1, 576, 7, 7]
            
#             # 2. 차원 변환
#             batch_size = features.size(0)
#             channel = features.size(1)
#             encoder_out = features.view(batch_size, channel, -1).permute(0, 2, 1)
            
#             # 3. 디코더 초기화
#             start_token = word_map['<start>']
#             input_word = torch.LongTensor([start_token]).to(image.device)
#             h = self.decoder.init_h(encoder_out.mean(dim=1))
            
#             seq = []
            
#             # 4. 단어 생성 루프
#             for i in range(max_len):
#                 embeddings = self.decoder.embedding(input_word)
#                 attention_weighted_encoding, _ = self.decoder.attention(encoder_out, h)
#                 gru_input = torch.cat([embeddings, attention_weighted_encoding], dim=1)
#                 h = self.decoder.decode_step(gru_input, h)
                
#                 preds = self.decoder.fc(h)
#                 predicted_id = preds.argmax(dim=1)
                
#                 seq.append(predicted_id.item())
#                 if rev_word_map[predicted_id.item()] == '<end>':
#                     break
#                 input_word = predicted_id
                
#             return [rev_word_map[k] for k in seq]
        
#껍데기
class Model(nn.Module):
    def __init__(self, vocab_size, width_mult=1.0, embed_dim=256, decoder_dim=512, attention_dim=256):
        super(Model, self).__init__()
        
        # [A] 인코더 설정 (MobileNetV3 Small)
        # width_mult를 조절해서 더 경량화 가능 (예: 0.5)
        gc.collect()
        
        print("Initializing Model with width_mult={}".format(width_mult))

        mobilenet = MobileNetV3Small(width_mult=width_mult)
        print("MobileNetV3 Small initialized.")
        # 마지막 분류기(classifier)를 떼어내고, 특징 추출기(features)만 가져옴
        self.encoder = mobilenet.features
        
        # 학습된 가중치 고정 (Fine-tuning 시에는 True로 변경)
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # Encoder 출력 채널 수 계산 (Small 기준: 576 * width_mult)
        # 기본(1.0)이면 576, 0.5면 288 등
        output_channels = int(576 * width_mult) 
        print("Encoder Output Channels (Hardcoded): {}".format(output_channels))

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

    def generate(self, image, word_map, rev_word_map, max_len=20, device=None):
        """
        실제 사용(Inference)을 위한 함수 (이미지 1장 -> 문장)
        """
        self.eval()
        with torch.no_grad():
            if device is not None:
                image = image.to(device)
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

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

# 1. Activation Functions (Torchvision 명칭 준수)
class SqueezeExcitation(nn.Module):
    def __init__(self, input_channels, squeeze_channels):
        super().__init__()
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        scale = x.mean((2, 3), keepdim=True)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.sigmoid(scale)
        return x * scale

# 2. InvertedResidual Block (Torchvision과 동일한 'block' 구조)
class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super().__init__()
        self.use_res_connect = stride == 1 and inp == oup

        layers = []
        activation_layer = nn.Hardswish if use_hs else nn.ReLU

        # Expansion
        if hidden_dim != inp:
            layers.append(nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                activation_layer(inplace=True)
            ))

        # Depthwise
        layers.append(nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            activation_layer(inplace=True)
        ))

        # Squeeze-and-Excitation
        if use_se:
            squeeze_channels = _make_divisible(hidden_dim // 4, 8)
            layers.append(SqueezeExcitation(hidden_dim, squeeze_channels))

        # Pointwise
        layers.append(nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        ))

        # 변수명 'block'이 torchvision 가중치 로드 시 핵심임
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        result = self.block(x)
        if self.use_res_connect:
            result += x
        return result

# 3. MobileNetV3 Main Class
class MobileNetV3Small(nn.Module):
    def __init__(self, num_classes=1000,width_mult=1.0):
        super().__init__()
        
        # features라는 이름으로 시퀀셜을 묶음 (torchvision 구조)
        input_channels = 16
        
        # [kernel, hidden_dim, oup, use_se, use_hs, stride]
        # torchvision.models.mobilenetv3small 사양 표
        bneck_conf = [
            [3, 16, 16, True, False, 2],
            [3, 72, 24, False, False, 2],
            [3, 88, 24, False, False, 1],
            [5, 96, 40, True, True, 2],
            [5, 240, 40, True, True, 1],
            [5, 240, 40, True, True, 1],
            [5, 120, 48, True, True, 1],
            [5, 144, 48, True, True, 1],
            [5, 288, 96, True, True, 2],
            [5, 576, 96, True, True, 1],
            [5, 576, 96, True, True, 1],
        ]

        # First Layer (Layer 0)
        layers = [nn.Sequential(
            nn.Conv2d(3, input_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.Hardswish(inplace=True)
        )]

        # Building InvertedResidual Blocks (Layer 1 ~ 11)
        for i, (k, exp, oup, se, hs, s) in enumerate(bneck_conf):
            layers.append(InvertedResidual(input_channels, exp, oup, k, s, se, hs))
            input_channels = oup
        
        # Last Convolution Layers (Layer 12, 13)
        last_conv_input = input_channels
        last_conv_output = _make_divisible(input_channels * 6, 8) # 576
        layers.append(nn.Sequential(
            nn.Conv2d(last_conv_input, last_conv_output, 1, 1, 0, bias=False),
            nn.BatchNorm2d(last_conv_output),
            nn.Hardswish(inplace=True)
        ))
        
        # torchvision은 'features'라는 이름으로 묶음
        self.features = nn.Sequential(*layers)

        # Classification Layer (Torchvision 명칭: classifier)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(last_conv_output, 1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x