import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

class Compose:
    """컴포지트 변환: 여러 변환을 순차적으로 적용"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class Resize:
    """이미지 크기 조정과 함께 바운딩 박스도 조정"""
    def __init__(self, height, width=None):
        # width가 None이면 height와 같은 값으로 설정 (정사각형)
        self.height = height
        self.width = width if width is not None else height
        self.size = (self.width, self.height)  # PIL은 (width, height) 순서 사용

    def __call__(self, image, target):
        orig_width, orig_height = image.size
        new_width, new_height = self.size
        
        # 이미지 리사이즈
        image = F.resize(image, self.size)
        
        if target is not None and len(target) > 0:
            # YOLO 형식: 이미 정규화된 좌표(0~1)를 사용하지만
            # 종횡비가 변경되는 경우 조정 필요
            width_ratio = new_width / orig_width
            height_ratio = new_height / orig_height
            
            # 타겟의 중심 x, 중심 y, 너비, 높이를 조정
            # YOLO 좌표는 [class_id, center_x, center_y, width, height]
            # class_id는 그대로 두고 나머지 좌표만 조정
            target_copy = target.clone()
            
            # YOLO 좌표는 정규화되어 있으므로, 종횡비 변화만 반영
            # 새 종횡비에 맞게 조정 (비율이 다른 경우)
            if width_ratio != height_ratio:
                # center_x 조정 (원래 정규화된 좌표를 역정규화 → 조정 후 다시 정규화)
                target_copy[:, 1] = target_copy[:, 1] * orig_width * width_ratio / new_width
                # center_y 조정
                target_copy[:, 2] = target_copy[:, 2] * orig_height * height_ratio / new_height
                # width 조정
                target_copy[:, 3] = target_copy[:, 3] * orig_width * width_ratio / new_width
                # height 조정
                target_copy[:, 4] = target_copy[:, 4] * orig_height * height_ratio / new_height
            
            target = target_copy
            
        return image, target

class ToTensor:
    """이미지를 텐서로 변환하고 타겟은 그대로 둠"""
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

class Normalize:
    """이미지 정규화"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

class RandomHorizontalFlip:
    """50% 확률로 이미지와 바운딩 박스를 수평으로 뒤집음"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if np.random.random() < self.prob:
            image = F.hflip(image)
            if target is not None and len(target) > 0:
                # YOLO 형식: [class_id, center_x, center_y, width, height]
                # x 좌표만 반전 (1.0 - x)
                target_copy = target.clone()
                target_copy[:, 1] = 1.0 - target_copy[:, 1]  # center_x 반전
                target = target_copy
                
        return image, target

class SpectrumDataset(Dataset):
    def __init__(self, img_dir, image_height, image_width, transform=None):
        self.img_dir = img_dir
        self.image_files = [f for f in os.listdir(img_dir) if f.endswith('.png') and '_marked' not in f]
        self.image_files.sort()  # 정렬하여 일관성 유지
        
        # 기본 변환 설정 (박스 좌표 조정 포함)
        if transform is None:
            self.transform = Compose([
                Resize(height=image_height, width=image_width),
                ToTensor(),
                # Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 이미지 로드
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        # 같은 이름의 txt 파일 찾기 (확장자만 변경)
        img_name = os.path.splitext(self.image_files[idx])[0]
        label_path = os.path.join(self.img_dir, f"{img_name}.txt")
        
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    values = line.strip().split()
                    if len(values) == 5:
                        # YOLO 형식: class_id x_center y_center width height
                        class_id = int(values[0])
                        x_center = float(values[1])
                        y_center = float(values[2])
                        width = float(values[3])
                        height = float(values[4])
                        
                        labels.append([class_id, x_center, y_center, width, height])
        
        # 라벨이 없는 경우 빈 텐서 생성
        labels_tensor = torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros((0, 5), dtype=torch.float32)
        
        # 이미지와 라벨에 변환 적용
        if self.transform:
            image, labels_tensor = self.transform(image, labels_tensor)
        
        return image, labels_tensor 