"""
데이터셋 클래스 및 데이터 로드 함수
"""
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from collections import defaultdict
from .config import TEST_IMAGE_DIR, CAPTIONS_FILE, get_image_transform

def encode_caption(caption, word_map, max_len=50):
    """캡션 텍스트를 정수 시퀀스로 변환"""
    tokens = caption.lower().split()
    encoded = [word_map.get('<start>', 1)]
    for token in tokens[:max_len-2]:
        encoded.append(word_map.get(token, word_map.get('<unk>', 3)))
    encoded.append(word_map.get('<end>', 2))
    while len(encoded) < max_len:
        encoded.append(word_map.get('<pad>', 0))
    return torch.LongTensor(encoded[:max_len])

class CaptionDataset(Dataset):
    """캡션 데이터셋 클래스"""
    def __init__(self, images_dir, captions_file, transform=None, word_map=None, max_len=50):
        self.images_dir = images_dir
        self.transform = transform
        self.word_map = word_map
        self.max_len = max_len
        
        available_images = set([f for f in os.listdir(images_dir) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        
        image_to_captions = defaultdict(list)
        if os.path.exists(captions_file):
            with open(captions_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                first_line = lines[0].strip() if lines else ""
                start_idx = 1 if first_line.lower().startswith('image') or first_line.lower().startswith('filename') else 0
                
                for line in lines[start_idx:]:
                    line = line.strip()
                    if not line:
                        continue
                    if ',' in line:
                        parts = line.split(',', 1)
                        if len(parts) == 2:
                            img_name = parts[0].strip()
                            caption = parts[1].strip()
                            if img_name and caption and img_name in available_images:
                                image_to_captions[img_name].append(caption)
        
        self.image_caption_pairs = []
        for img_name, captions in image_to_captions.items():
            if captions:
                for caption in captions:
                    self.image_caption_pairs.append((img_name, caption))
    
    def __getitem__(self, idx):
        img_name, caption_text = self.image_caption_pairs[idx]
        img_path = os.path.join(self.images_dir, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception:
            image = torch.zeros(3, 224, 224)
        
        if self.word_map:
            caption = encode_caption(caption_text, self.word_map, self.max_len)
        else:
            caption = torch.zeros(self.max_len, dtype=torch.long)
        return image, caption
    
    def __len__(self):
        return len(self.image_caption_pairs)

def load_test_data(device, transform=None, test_image_dir=None, captions_file=None):
    """테스트 이미지와 참조 캡션 로드
    
    Args:
        device: torch device
        transform: 이미지 transform (None이면 기본 transform 사용)
        test_image_dir: 테스트 이미지 디렉토리 (None이면 기본 경로 사용)
        captions_file: 캡션 파일 경로 (None이면 기본 경로 사용)
    
    Returns:
        img_tensor, ref_caption
    """
    if transform is None:
        transform = get_image_transform()
    if test_image_dir is None:
        test_image_dir = TEST_IMAGE_DIR
    if captions_file is None:
        captions_file = CAPTIONS_FILE
    
    img_tensor = None
    filename = None
    ref_caption = None
    
    if os.path.exists(test_image_dir):
        files = [f for f in os.listdir(test_image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if files:
            import random
            filename = random.choice(files)
            img_path = os.path.join(test_image_dir, filename)
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            print("📸 테스트 이미지: {}".format(filename))
    
    if img_tensor is None:
        print("⚠️ 이미지를 찾을 수 없어 더미 데이터를 사용합니다.")
        img_tensor = torch.randn(1, 3, 224, 224).to(device)
        filename = "dummy"
        ref_caption = "a test image"
    else:
        # 참조 캡션 로드
        if os.path.exists(captions_file) and filename != "dummy":
            with open(captions_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    if ',' in line:
                        parts = line.split(',', 1)
                        if len(parts) == 2 and parts[0].strip() == filename:
                            ref_caption = parts[1].strip()
                            print("📝 참조 캡션: {}".format(ref_caption))
                            break
    
    return img_tensor, ref_caption

def prepare_calibration_dataset(word_map, num_samples=100, test_image_dir=None, transform=None):
    """정적 양자화를 위한 Calibration 데이터셋 준비
    
    Args:
        word_map: 단어 맵
        num_samples: 샘플 개수
        test_image_dir: 테스트 이미지 디렉토리 (None이면 기본 경로 사용)
        transform: 이미지 transform (None이면 기본 transform 사용)
    
    Returns:
        calibration_images, calibration_captions
    """
    if transform is None:
        transform = get_image_transform()
    if test_image_dir is None:
        test_image_dir = TEST_IMAGE_DIR
    
    calibration_images = []
    calibration_captions = []
    
    if not os.path.exists(test_image_dir):
        print("   ⚠️ 이미지 디렉토리가 없어 더미 데이터를 사용합니다.")
        for _ in range(num_samples):
            dummy_img = torch.randn(1, 3, 224, 224)
            calibration_images.append(dummy_img)
            dummy_cap = torch.LongTensor([
                word_map.get('<start>', 1),
                word_map.get('<pad>', 0),
                word_map.get('<end>', 2)
            ])
            calibration_captions.append(dummy_cap)
        return calibration_images, calibration_captions
    
    image_files = [f for f in os.listdir(test_image_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print("   ⚠️ 이미지가 없어 더미 데이터를 사용합니다.")
        for _ in range(num_samples):
            dummy_img = torch.randn(1, 3, 224, 224)
            calibration_images.append(dummy_img)
            dummy_cap = torch.LongTensor([
                word_map.get('<start>', 1),
                word_map.get('<pad>', 0),
                word_map.get('<end>', 2)
            ])
            calibration_captions.append(dummy_cap)
        return calibration_images, calibration_captions
    
    import random
    selected_files = random.sample(image_files, min(num_samples, len(image_files)))
    
    print("   📊 Calibration 데이터셋 준비 중: {}개 이미지".format(len(selected_files)))
    
    for filename in selected_files:
        try:
            img_path = os.path.join(test_image_dir, filename)
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)
            calibration_images.append(img_tensor)
            
            dummy_cap = torch.LongTensor([
                word_map.get('<start>', 1),
                word_map.get('<pad>', 0),
                word_map.get('<end>', 2)
            ])
            calibration_captions.append(dummy_cap)
        except Exception as e:
            print("   ⚠️ 이미지 로드 실패 ({}): {}".format(filename, e))
            continue
    
    while len(calibration_images) < num_samples:
        dummy_img = torch.randn(1, 3, 224, 224)
        calibration_images.append(dummy_img)
        dummy_cap = torch.LongTensor([
            word_map.get('<start>', 1),
            word_map.get('<pad>', 0),
            word_map.get('<end>', 2)
        ])
        calibration_captions.append(dummy_cap)
    
    return calibration_images[:num_samples], calibration_captions[:num_samples]

