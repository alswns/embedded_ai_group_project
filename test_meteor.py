#!/usr/bin/env python3
"""
METEOR 점수 계산 테스트
"""

from src.utils.metrics import calculate_meteor

# 테스트 케이스
test_cases = [
    {
        "generated": "a large brown dog is pushing a small boston terrier with a ball <end>",
        "reference": "a large brown dog is pushing a small dog",
        "description": "캡션 + <end> 토큰 포함"
    },
    {
        "generated": "a man and a woman sitting on a dock <end>",
        "reference": "a man and a woman are sitting on a dock",
        "description": "캡션 + <end> 토큰"
    },
    {
        "generated": "a cat on the mat",
        "reference": "a cat is on the mat",
        "description": "일반 캡션"
    },
    {
        "generated": "<start> a dog is running <end>",
        "reference": "a dog is running fast",
        "description": "<start>, <end> 토큰 포함"
    }
]

print("=" * 70)
print("METEOR 점수 계산 테스트")
print("=" * 70)

for i, test in enumerate(test_cases, 1):
    print(f"\n테스트 {i}: {test['description']}")
    print(f"  Generated: {test['generated']}")
    print(f"  Reference: {test['reference']}")
    
    score = calculate_meteor(test['generated'], test['reference'])
    
    print(f"  ✅ METEOR Score: {score:.4f}")
    
    # 점수 범위 확인
    if 0.0 <= score <= 1.0:
        print(f"  ✓ 점수 범위 정상 (0.0~1.0)")
    else:
        print(f"  ✗ 점수 범위 오류! ({score})")

print("\n" + "=" * 70)
print("테스트 완료!")
print("=" * 70)
