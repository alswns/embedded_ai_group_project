#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jetson Nano용 최소화된 테스트 스크립트
main 함수 실행 전 모듈 로드 검증
"""

import sys
import os

print("=" * 70, file=sys.stderr)
print("📦 모듈 로드 검증 시작", file=sys.stderr)
print("=" * 70, file=sys.stderr)

# 1. 기본 모듈 로드
print("\n1️⃣  기본 모듈 로드...", file=sys.stderr)
try:
    import cv2
    print("   ✅ cv2 로드 완료", file=sys.stderr)
    
    import torch
    print("   ✅ torch 로드 완료", file=sys.stderr)
    
    import numpy as np
    print("   ✅ numpy 로드 완료", file=sys.stderr)
    
    import psutil
    print("   ✅ psutil 로드 완료", file=sys.stderr)
    
    import gc
    print("   ✅ gc 로드 완료", file=sys.stderr)
except Exception as e:
    print("   ❌ 기본 모듈 로드 실패: {}".format(e), file=sys.stderr)
    sys.exit(1)

# 2. 심화 모듈 로드
print("\n2️⃣  심화 모듈 로드...", file=sys.stderr)
try:
    from PIL import Image
    print("   ✅ PIL 로드 완료", file=sys.stderr)
    
    from torchvision import transforms
    print("   ✅ torchvision 로드 완료", file=sys.stderr)
except Exception as e:
    print("   ⚠️  시각화 라이브러리 로드 실패: {}".format(e), file=sys.stderr)

# 3. 프로젝트 모듈 로드
print("\n3️⃣  프로젝트 모듈 로드...", file=sys.stderr)
try:
    from src.muti_modal_model.model import MobileNetCaptioningModel
    print("   ✅ MobileNetCaptioningModel 로드 완료", file=sys.stderr)
    
    from src.utils.quantization_utils import apply_dynamic_quantization
    print("   ✅ quantization_utils 로드 완료", file=sys.stderr)
except Exception as e:
    print("   ❌ 프로젝트 모듈 로드 실패: {}".format(e), file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. 환경 설정 검증
print("\n4️⃣  환경 설정 검증...", file=sys.stderr)
try:
    # GPU 비활성화
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    
    # CPU 설정
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    
    device = torch.device("cpu")
    print("   ✅ CPU 모드 설정 완료", file=sys.stderr)
    print("   📍 디바이스: {}".format(device), file=sys.stderr)
except Exception as e:
    print("   ❌ 환경 설정 실패: {}".format(e), file=sys.stderr)
    sys.exit(1)

print("\n" + "=" * 70, file=sys.stderr)
print("✅ 모든 모듈 로드 성공!", file=sys.stderr)
print("=" * 70, file=sys.stderr)
print("\n이제 run.py를 실행하세요:")
print("  python3 scripts/run.py\n")
