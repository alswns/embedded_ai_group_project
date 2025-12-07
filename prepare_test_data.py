"""
Test 데이터셋 준비 (100장의 이미지 + 캡션)
"""
import os
import shutil
from pathlib import Path

# 경로 설정
assets_dir = '/Users/bagminjun/Desktop/임베디드/assets'
images_dir = os.path.join(assets_dir, 'Images')
captions_file = os.path.join(assets_dir, 'captions.txt')
test_dir = '/Users/bagminjun/Desktop/임베디드/test'

# test 폴더 생성
os.makedirs(test_dir, exist_ok=True)
os.makedirs(os.path.join(test_dir, 'images'), exist_ok=True)

# 캡션 파일 읽기
captions_data = {}
with open(captions_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # 형식: image_id.jpg#cap_num\tcaption
        parts = line.split(',')
        if len(parts) >= 2:
            img_info, caption = parts[0], '\t'.join(parts[1:])
            img_name = img_info.split('#')[0]
            
            if img_name not in captions_data:
                captions_data[img_name] = []
            captions_data[img_name].append(caption)

print(f"Total unique images: {len(captions_data)}")

# 이미지 복사 및 캡션 저장
image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
print(f"Total image files: {len(image_files)}")

# 처음 100개만 선택
selected_images = image_files[:100]

captions_output = os.path.join(test_dir, 'captions.txt')
with open(captions_output, 'w', encoding='utf-8') as out_f:
    for idx, img_file in enumerate(selected_images):
        src = os.path.join(images_dir, img_file)
        dst = os.path.join(test_dir, 'images', img_file)
        
        # 이미지 복사
        shutil.copy2(src, dst)
        # 캡션 저장 (첫번째 캡션만)
        # print(f"Processing {img_file}...", end=' ')
        # print(captions_data.keys())
        if img_file in captions_data.keys() and captions_data[img_file]:
            caption = captions_data[img_file][0]
            print(img_file)
            out_f.write(f"{img_file},{caption}\n")
        
        if (idx + 1) % 20 == 0:
            print(f"Processed {idx + 1}/100 images")

print(f"\nTest dataset ready!")
print(f"- Images: {os.path.join(test_dir, 'images')}")
print(f"- Captions: {captions_output}")
