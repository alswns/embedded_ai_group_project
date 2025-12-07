#!/bin/bash
# Jetson Nano 최적화 실행 스크립트

echo "======================================================================"
echo "       Jetson Nano 이미지 캡셔닝 시스템"
echo "======================================================================"
echo ""

# 메모리 확인
TOTAL_MEM=$(free -h | grep Mem | awk '{print $2}')
USED_MEM=$(free -h | grep Mem | awk '{print $3}')
AVAIL_MEM=$(free -h | grep Mem | awk '{print $7}')

echo "📊 시스템 정보"
echo "   총 메모리: $TOTAL_MEM"
echo "   사용 중: $USED_MEM"
echo "   가용: $AVAIL_MEM"
echo ""

# 메모리 충분성 확인
AVAIL_MB=$(free | grep Mem | awk '{print $7}')
if [ "$AVAIL_MB" -lt 1500 ]; then
    echo "⚠️  경고: 메모리가 부족합니다 ($AVAIL_MB MB)"
    echo "   권장: 최소 1500MB 이상"
    echo ""
    read -p "계속 실행하시겠습니까? (y/n): " response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "취소되었습니다."
        exit 1
    fi
fi

echo "🚀 권장 구성:"
echo "   모델: Pruned Model"
echo "   양자화: FP16 또는 INT8"
echo ""

# Python 버전 확인
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "🐍 Python 버전: $PYTHON_VERSION"
echo ""

# PyTorch 확인
echo "🔍 PyTorch 및 CUDA 확인 중..."
python3 << 'PYEOF'
import torch
print("   PyTorch 버전: {}".format(torch.__version__))
print("   CUDA 사용 가능: {}".format(torch.cuda.is_available()))
if torch.cuda.is_available():
    print("   GPU: {}".format(torch.cuda.get_device_name(0)))
    print("   GPU 메모리: {}MB".format(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024))
else:
    print("   디바이스: CPU")
PYEOF

echo ""
echo "======================================================================"
echo "                     프로그램 시작"
echo "======================================================================"
echo ""

# 실행
python3 -m scripts.run

# 종료 후 정리
echo ""
echo "======================================================================"
echo "                     프로그램 종료"
echo "======================================================================"
