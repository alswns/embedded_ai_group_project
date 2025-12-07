
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
        

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

# 2. Activation Functions (Hard-Sigmoid, Hard-Swish)
class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    def forward(self, x):
        return self.relu(x + 3) / 6

class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.sigmoid = Hsigmoid(inplace=inplace)
    def forward(self, x):
        return x * self.sigmoid(x)

# 3. Squeeze-and-Excitation (SE) Block
# 채널 간의 중요도를 학습하여 성능을 높이는 모듈
class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, _make_divisible(channel // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(_make_divisible(channel // reduction, 8), channel),
            Hsigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# 4. Inverted Residual Block (MobileNet의 핵심 빌딩 블록)
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, exp, kernel, se, act):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        activation = Hswish if act == 'HS' else nn.ReLU

        # Expansion (1x1 conv로 채널 뻥튀기)
        if exp != inp:
            layers.append(nn.Sequential(
                nn.Conv2d(inp, exp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(exp),
                activation(inplace=True)
            ))
        
        # Depthwise Conv (채널별 연산)
        layers.append(nn.Sequential(
            nn.Conv2d(exp, exp, kernel, stride, (kernel-1)//2, groups=exp, bias=False),
            nn.BatchNorm2d(exp),
            activation(inplace=True)
        ))

        # SE Block (선택적 적용)
        if se:
            layers.append(SELayer(exp))

        # Pointwise Linear Conv (다시 채널 줄이기, 활성화 함수 없음)
        layers.append(nn.Sequential(
            nn.Conv2d(exp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

# 5. MobileNetV3-Small 메인 클래스
class MobileNetV3Small(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0):
        super(MobileNetV3Small, self).__init__()
        
        # 설정값: [kernel, exp_size, out_channels, use_se, activation, stride]
        # 논문의 MobileNetV3-Small 사양표 그대로 구현
        cfg = [
            # k, exp, out,  se,     nl,  s 
            [3, 16,  16,  True,  'RE', 2],
            [3, 72,  24,  False, 'RE', 2],
            [3, 88,  24,  False, 'RE', 1],
            [5, 96,  40,  True,  'HS', 2],
            [5, 240, 40,  True,  'HS', 1],
            [5, 240, 40,  True,  'HS', 1],
            [5, 120, 48,  True,  'HS', 1],
            [5, 144, 48,  True,  'HS', 1],
            [5, 288, 96,  True,  'HS', 2],
            [5, 576, 96,  True,  'HS', 1],
            [5, 576, 96,  True,  'HS', 1],
        ]

        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            Hswish(inplace=True)
        )]

        # 블록 쌓기
        for k, exp, out, se, nl, s in cfg:
            output_channel = _make_divisible(out * width_mult, 8)
            exp_channel = _make_divisible(exp * width_mult, 8)
            layers.append(InvertedResidual(input_channel, output_channel, s, exp_channel, k, se, nl))
            input_channel = output_channel
        
        # 마지막 Conv (Features 단계)
        last_conv_channel = _make_divisible(576 * width_mult, 8)
        layers.append(nn.Sequential(
            nn.Conv2d(input_channel, last_conv_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(last_conv_channel),
            Hswish(inplace=True)
        ))
        
        self.features = nn.Sequential(*layers)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(last_conv_channel, 1024),
            Hswish(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes),
        )

        # 가중치 초기화
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)  # [Batch, 576, 7, 7] (입력이 224일 때)
        x = self.avgpool(x)   # [Batch, 576, 1, 1]
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)