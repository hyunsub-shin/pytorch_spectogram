import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import SpectrumDataset, Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip
from models.model_MobileNetV3 import SpectrumModel
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import platform  # 폰트관련 운영체제 확인
from PIL import Image, ImageDraw, ImageFont
import sys

# 클래스 이름 지정
CLASS_NAMES = ["WiFi", "collision", "BT"]  
# 클래스별 색상 정의 (R, G, B) - 전역 변수로 설정
CLASS_COLORS = {
    0: (0, 255, 0),     # WiFi: 녹색
    1: (255, 0, 0),     # collision: 빨간색
    2: (0, 0, 255)      # BT: 파란색
}

# 폰트 파일 경로 설정 (운영체제별)
if platform.system() == 'Windows':
    # Windows의 경우 시스템 폰트 경로 사용
    FONT_PATH = os.path.join(os.environ['SYSTEMROOT'], 'Fonts', 'malgun.ttf')
    FONT_NAME = 'Malgun Gothic'
elif platform.system() == 'Darwin':  # macOS
    FONT_PATH = '/System/Library/Fonts/AppleGothic.ttf'
    FONT_NAME = 'AppleGothic'
else:  # Linux 등
    # 일반적인 Linux 폰트 경로들
    possible_font_paths = [
        '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
        '/usr/share/fonts/nanum/NanumGothic.ttf',
        '/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf'
    ]
    FONT_PATH = None
    for path in possible_font_paths:
        if os.path.exists(path):
            FONT_PATH = path
            break
    if FONT_PATH is None:
        FONT_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'  # 폴백 폰트
    FONT_NAME = 'NanumGothic'

# 디버그 출력
print(f"OS: {platform.system()}")
print(f"폰트 경로: {FONT_PATH}")
print(f"폰트가 존재함: {os.path.exists(FONT_PATH)}")

# Matplotlib 폰트 설정
try:
    plt.rc('font', family=FONT_NAME)
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"Matplotlib 폰트 설정 오류: {e}")


def visualize_samples_check(img_dir, num_samples=2):
    """
    데이터셋에서 샘플 이미지와 라벨을 시각화하는 함수
    Args:
        img_dir: 이미지 디렉토리 경로
        num_samples: 표시할 샘플 수
    """
    # 이미지 파일 리스트 가져오기
    image_files = [f for f in os.listdir(img_dir) if f.endswith('.png') and '_marked' not in f]
    image_files.sort()  # 정렬하여 일관성 유지

    # 샘플 수 제한
    if num_samples > len(image_files):
        num_samples = len(image_files)

    # 클래스별 색상 정의 (RGB 형식)
    colors = CLASS_COLORS

    for i in range(num_samples):
        img_path = os.path.join(img_dir, image_files[i])
        image = Image.open(img_path).convert('RGB')

        # 같은 이름의 txt 파일 찾기 (확장자만 변경)
        img_name = os.path.splitext(image_files[i])[0]
        label_path = os.path.join(img_dir, f"{img_name}.txt")

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

        # 라벨이 없는 경우 빈 리스트
        labels_tensor = torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros((0, 5), dtype=torch.float32)

        # 이미지 시각화
        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.title(f"Image: {image_files[i]}")
        plt.axis('off')

        # 라벨 시각화
        for label in labels_tensor:
            class_id, x_center, y_center, width, height = label
            # YOLO 좌표를 이미지 좌표로 변환
            x1 = (x_center - width / 2) * image.width
            y1 = (y_center - height / 2) * image.height
            x2 = (x_center + width / 2) * image.width
            y2 = (y_center + height / 2) * image.height
            
            # 바운딩 박스 그리기 (클래스별 색상 사용, 0-1 범위로 변환)
            color = tuple(c / 255.0 for c in colors[int(class_id)])  # 0-255를 0-1로 변환
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor=color, facecolor='none', linewidth=2))

        plt.show()


# 폰트 로드 함수
def load_font(size=12):
    """지정된 크기의 폰트를 로드하는 함수
    
    Args:
        size: 폰트 크기
        
    Returns:
        ImageFont: 로드된 폰트 객체
    """
    try:
        # 1. 지정된 폰트 파일 경로로 로드 시도
        if os.path.exists(FONT_PATH):
            return ImageFont.truetype(FONT_PATH, size)
        
        # 2. 시스템에 설치된 폰트 이름으로 로드 시도
        return ImageFont.truetype(FONT_NAME, size)
    except Exception as e:
        print(f"폰트 로드 오류: {e}")
        try:
            # Windows 대체 폰트
            if platform.system() == 'Windows':
                windows_fonts = [
                    os.path.join(os.environ['SYSTEMROOT'], 'Fonts', 'gulim.ttc'),  # 굴림체
                    os.path.join(os.environ['SYSTEMROOT'], 'Fonts', 'batang.ttc'),  # 바탕체
                    os.path.join(os.environ['SYSTEMROOT'], 'Fonts', 'arial.ttf')   # 영문 폰트
                ]
                for font_path in windows_fonts:
                    if os.path.exists(font_path):
                        return ImageFont.truetype(font_path, size)
            
            # macOS 대체 폰트
            elif platform.system() == 'Darwin':
                for mac_font in ['/Library/Fonts/AppleGothic.ttf', '/Library/Fonts/Arial.ttf']:
                    if os.path.exists(mac_font):
                        return ImageFont.truetype(mac_font, size)
            
            # 기본 폰트 시도
            return ImageFont.load_default()
        except Exception:
            print("모든 폰트 로드 시도 실패, 기본 폰트 사용")
            return ImageFont.load_default()


def collate_fn(batch):
    """가변 길이 라벨을 처리하기 위한 collate_fn 정의
    
    Args:
        batch: 데이터로더에서 로드된 배치
        
    Returns:
        images: 이미지 텐서 배치
        labels: 라벨 배치
    """
    images = []
    labels = []
    
    for img, label in batch:
        images.append(img)
        labels.append(label)
    
    images = torch.stack(images)
    return images, labels


def evaluate():
    """mAP 평가 함수 (간소화)
    
    향후 구현을 위한 플레이스홀더 함수
    
    Returns:
        float: 평가 점수 (mAP)
    """
    model.eval()
    return 0.0  # 실제 구현에서는 mAP 계산


def visualize_dataset_samples(dataset, num_samples=2):
    """데이터셋에서 샘플 이미지를 불러와 바운딩 박스와 함께 시각화
    
    Args:
        dataset: 시각화할 데이터셋
        num_samples: 시각화할 샘플 수
    """
    # 저장 디렉토리 생성
    os.makedirs('debug', exist_ok=True)
    
    # 클래스별 색상 정의 (RGB 형식)
    colors = CLASS_COLORS    
    class_names = CLASS_NAMES
    
    # 원본 이미지 비율 설정 (좌우로 길게)
    display_width = 1024
    display_height = 192
    
    for i in range(min(num_samples, len(dataset))):
        # 데이터셋에서 이미지와 레이블 가져오기
        image_tensor, targets = dataset[i]
        
        # 이미지 텐서를 PIL 이미지로 변환 (시각화용)
        image_np = image_tensor.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
        
        # RGB 이미지로 변환 (값 범위: 0-255)
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)
        
        # 원본 이미지 비율로 새 이미지 생성 (좌우로 길게)
        pil_image = Image.fromarray(image_np).resize((display_width, display_height), Image.LANCZOS)
        
        # 이미지 크기 가져오기
        width, height = pil_image.size
        print(f"시각화 이미지 크기: {width} x {height} (좌우 비율: {width/height:.1f}:1)")
        
        draw = ImageDraw.Draw(pil_image)
        
        # 개선된 폰트 로딩 함수 사용
        font = load_font(12)
        
        # 바운딩 박스 그리기
        for target in targets:
            # 타겟의 형식: [class_id, x_center, y_center, width, height]
            class_id = int(target[0])
            x_center = float(target[1]) * width
            y_center = float(target[2]) * height
            bbox_width = float(target[3]) * width
            bbox_height = float(target[4]) * height
            
            # 박스 좌표 계산 (중심 -> 좌상단, 우하단)
            x1 = max(0, int(x_center - bbox_width / 2))
            y1 = max(0, int(y_center - bbox_height / 2))
            x2 = min(width, int(x_center + bbox_width / 2))
            y2 = min(height, int(y_center + bbox_height / 2))
            
            # 박스 그리기
            draw.rectangle([x1, y1, x2, y2], outline=colors[class_id], width=2)
            
            # 레이블 텍스트
            label_text = f"{class_names[class_id]}"
            
            # 텍스트 배경 그리기
            text_size = draw.textbbox((0, 0), label_text, font=font)
            text_width = text_size[2] - text_size[0]
            text_height = text_size[3] - text_size[1]
            
            draw.rectangle(
                [x1, y1 - text_height - 2, x1 + text_width + 2, y1],
                fill=colors[class_id]
            )
            
            # 텍스트 그리기
            draw.text((x1 + 1, y1 - text_height - 1), label_text, fill=(255, 255, 255), font=font)
            
        # 이미지에 총 객체 수 표시 (한글 텍스트)
        info_text = f"이미지 {i+1}: 총 {len(targets)}개 객체"
        
        # 한글 텍스트 배경 만들기
        text_bbox = draw.textbbox((0, 0), info_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # 텍스트 배경 그리기
        draw.rectangle(
            [10, 10, 10 + text_width + 4, 10 + text_height + 4],
            fill=(0, 0, 0, 128)  # 반투명 검정 배경
        )
        
        # 한글 텍스트 그리기
        draw.text((12, 12), info_text, fill=(255, 255, 0), font=font)
        
        # 원본 비율로 저장할 이미지 생성 (좌우로 길게)
        figure = plt.figure(figsize=(20, 4))  # 좌우로 긴 비율 설정
        plt.imshow(np.array(pil_image))
        plt.axis('off')  # 축 표시 제거
        
        # # 이미지 저장
        # output_path = f"debug/sample_{i+1}_boxes.png"
        # plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=150)
        # plt.close()
        # print(f"디버그 이미지 저장됨: {output_path}")
        
        # 추가로 PIL 이미지도 직접 저장 (비교용)
        pil_output_path = f"debug/sample_{i+1}_boxes_direct.png"
        pil_image.save(pil_output_path)
        
        # 콘솔에 타겟 정보 출력
        print(f"이미지 {i+1} 타겟 수: {len(targets)}")
        for t_idx, target in enumerate(targets[:5]):  # 처음 5개만 출력
            class_id = int(target[0])
            print(f"  타겟 {t_idx+1}: 클래스={class_names[class_id]}, 좌표=(x={target[1]:.3f}, y={target[2]:.3f}, w={target[3]:.3f}, h={target[4]:.3f})")
        if len(targets) > 5:
            print(f"  ... 외 {len(targets) - 5}개")


def train_dataset_and_loader():
    """학습 데이터셋과 데이터 로더 생성
    
    Returns:
        DataLoader: 학습 데이터 로더
    """
    # 커스텀 변환 클래스 사용
    transform = Compose([
        Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
        # RandomHorizontalFlip(0.5),  # 데이터 증강: 50% 확률로 좌우 반전
        ToTensor(),
        # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 데이터셋 생성
    train_dataset = SpectrumDataset(
        img_dir=TRAIN_DIR,
        image_height=IMG_HEIGHT, 
        image_width=IMG_WIDTH,
        transform=transform
    )
    
    # 데이터로더 생성
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,  # 학습 데이터 셔플
        collate_fn=collate_fn  # 가변 길이 라벨을 처리하기 위한 함수
    )
    
    print(f"훈련 데이터셋 크기: {len(train_dataset)}개 이미지")
    return train_dataset, train_loader


def val_dataset_and_loader():
    """검증 데이터셋과 데이터 로더 생성
    
    Returns:
        DataLoader: 검증 데이터 로더
    """
    # 커스텀 변환 클래스 사용 (데이터 증강 없음)
    transform = Compose([
        Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
        ToTensor(),
        # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 데이터셋 생성
    val_dataset = SpectrumDataset(
        img_dir=VAL_DIR,
        image_height=IMG_HEIGHT, 
        image_width=IMG_WIDTH,
        transform=transform
    )
    
    # 데이터로더 생성
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,  # 검증 데이터는 셔플하지 않음
        collate_fn=collate_fn
    )
    
    print(f"검증 데이터셋 크기: {len(val_dataset)}개 이미지")
    return val_dataset, val_loader


def create_model():
    """모델 및 옵티마이저, 스케줄러 생성
    
    Returns:
        tuple: (모델, 옵티마이저, 스케줄러) 튜플
    """
    # 모델 생성 (MobileNetV3 사용)
    model = SpectrumModel(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
    
    # 옵티마이저 생성
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 학습률 스케줄러 생성 (손실이 특정 에폭 동안 개선되지 않으면 학습률 감소)
    # verbose 파라미터 제거 (deprecated 경고 해결)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        patience=5, 
        factor=0.1
    )
    
    return model, optimizer, scheduler


def save_checkpoint(model, optimizer, epoch, loss, path):
    """모델 체크포인트 저장
    
    Args:
        model: 저장할 모델
        optimizer: 저장할 옵티마이저
        epoch: 현재 에폭
        loss: 현재 손실 값
        path: 저장 경로
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def plot_loss(train_losses, val_losses, path):
    """학습 및 검증 손실 그래프 생성 및 저장
    
    Args:
        train_losses: 학습 손실 리스트
        val_losses: 검증 손실 리스트
        path: 그래프 저장 경로
    """
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='학습 손실')
    plt.plot(epochs, val_losses, 'r-', label='검증 손실')
    
    plt.xlabel('에폭')
    plt.ylabel('손실')
    plt.title('학습 및 검증 손실 그래프')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(path)
    plt.close()


def get_lr(optimizer):
    """현재 학습률 반환
    
    Args:
        optimizer: 옵티마이저
        
    Returns:
        float: 현재 학습률
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_epoch(model, train_loader, optimizer, device):
    """한 에폭 동안의 학습 수행
    
    Args:
        model: 학습할 모델
        train_loader: 학습 데이터 로더
        optimizer: 옵티마이저
        device: 학습 장치 (CPU/GPU)
        
    Returns:
        float: 에폭 손실 값
    """
    model.train()
    running_loss = 0.0
    
    # 프로그레스 바로 학습 진행 상황 표시
    progress_bar = tqdm(train_loader, desc=f"학습 중", leave=False)
    
    for batch_idx, (images, targets) in enumerate(progress_bar):
        # 이미지를 장치로 이동
        images = images.to(device)
        
        # 손실 계산 및 역전파
        optimizer.zero_grad()
        outputs = model(images)
        
        # 모델의 손실 계산 함수 사용
        loss = model.compute_loss(outputs, targets, device)
        loss.backward()
        
        # 그래디언트 클리핑 (폭발적인 그래디언트 방지)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        
        optimizer.step()
        
        # 손실 누적 및 진행 상황 업데이트
        running_loss += loss.item()
        progress_bar.set_postfix(loss=running_loss / (batch_idx + 1))
    
    # 에폭 평균 손실 계산
    epoch_loss = running_loss / len(train_loader)

    return epoch_loss


def validation_epoch(model, val_loader, device):
    """한 에폭 동안의 검증 수행
    
    Args:
        model: 검증할 모델
        val_loader: 검증 데이터 로더
        device: 학습 장치 (CPU/GPU)
        
    Returns:
        float: 검증 손실 값
    """
    model.eval()  # 모델을 평가 모드로 설정
    running_loss = 0.0
    
    # 진행 상황 표시 프로그레스 바
    progress_bar = tqdm(val_loader, desc=f"검증 중", leave=False)
    
    # 그래디언트 계산 비활성화
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(progress_bar):
            # 이미지를 장치로 이동
            images = images.to(device)
            
            # 순전파
            outputs = model(images)
            
            # 손실 계산
            loss = model.compute_loss(outputs, targets, device)
            
            # 손실 누적 및 진행 상황 업데이트
            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (batch_idx + 1))
    
    # 검증 평균 손실 계산
    val_loss = running_loss / len(val_loader)
    
    return val_loss


def main():
    # 결과 저장 디렉토리 생성
    os.makedirs('results', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    """메인 학습 루프"""
    if DEVICE.type == 'cuda':        
        print(f'현재 사용 중인 GPU: {torch.cuda.get_device_name(0)}')
        
        # CUDA 설정
        torch.backends.cudnn.benchmark = True # 속도 향상을 위한 설정
        torch.backends.cudnn.deterministic = True # 재현 가능성 확보
        torch.backends.cuda.matmul.allow_tf32 = True  # TensorFloat-32 사용        
        torch.cuda.empty_cache()
        # 메모리 할당 모드 설정
        # torch.cuda.set_per_process_memory_fraction(0.8)  # GPU 메모리의 80% 사용

    print("\n=== 시스템 정보 ===")
    print(f"CUDA 사용 가능 여부: {torch.cuda.is_available()}")
    print(f"현재 PyTorch의 CUDA 버전: {torch.version.cuda}")
    print(f"PyTorch 버전: {torch.__version__}") # 예: '2.0.0+cu121'는 CUDA 12.1 버전을 지원
    print(f'사용 중인 장치: {DEVICE}')
    print('-' * 50)
        
    # 데이터로더 생성
    train_dataset, train_loader = train_dataset_and_loader()
    val_dataset, val_loader = val_dataset_and_loader()
    
    #######################################################################
    # # 데이터셋 시각화 (디버깅용)
    # print("\n=== 데이터셋 시각화 (디버깅) ===")
    # print("학습 데이터셋 샘플 시각화:")
    # visualize_dataset_samples(train_dataset, num_samples=5)
    
    # # print("\n검증 데이터셋 샘플 시각화:")
    # # visualize_dataset_samples(val_dataset, num_samples=5)
    # print('-' * 50)
    
    # user_input = input("데이터셋 시각화 확인 후 학습을 계속하려면 'y'를 입력하세요 (다른 키 입력 시 종료): ")
    # if user_input.lower() != 'y':
    #     print("학습을 종료합니다.")
    #     return
    #######################################################################
    
    # 모델, 옵티마이저, 스케줄러 생성
    global model  # evaluate 함수에서 사용하기 위해 전역 변수로 설정
    model, optimizer, scheduler = create_model()
    
    # 손실 기록
    train_losses = []
    val_losses = []
    # 학습률 기록
    lr_history = []
    
    # 최적 모델 추적을 위한 변수
    best_val_loss = float('inf')
    
    # 학습 루프
    for epoch in range(EPOCHS):
        # 현재 학습률 저장
        current_lr = get_lr(optimizer)
        lr_history.append(current_lr)
        
        print(f"\n에폭 {epoch+1}/{EPOCHS} (학습률: {current_lr:.9f})")
        
        # 한 에폭 학습 수행
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
        train_losses.append(train_loss)
        
        # 검증 수행
        val_loss = validation_epoch(model, val_loader, DEVICE)
        val_losses.append(val_loss)
        
        print(f"학습 손실: {train_loss:.9f}, 검증 손실: {val_loss:.9f}")
        
        # 최적 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss, 
                'checkpoints/best_spectrum_model.pth'
            )
            print(f"새로운 최적 모델 저장 (검증 손실: {val_loss:.9f})")
        
        # LR 스케줄러 업데이트 (검증 손실 기반)
        prev_lr = get_lr(optimizer)
        scheduler.step(val_loss)
        current_lr = get_lr(optimizer)
        
        # 학습률 변경 확인 및 출력
        if prev_lr != current_lr:
            print(f"학습률 변경: {prev_lr:.9f} -> {current_lr:.9f}")
        
        # 모델 정기 저장 (5 에폭마다 또는 마지막 에폭에)
        if (epoch + 1) % 5 == 0 or (epoch + 1) == EPOCHS:
            checkpoint_path = f'checkpoints/spectrum_model_epoch_{epoch+1}.pth'
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
    
    # 손실 그래프 그리기
    plot_loss(train_losses, val_losses, 'results/loss_curve.png')
    
    # # 학습률 변화 그래프 그리기
    # plt.figure(figsize=(10, 5))
    # plt.plot(range(1, len(lr_history) + 1), lr_history, marker='o', label='학습률')
    # plt.xlabel('에폭')
    # plt.ylabel('학습률')
    # plt.title('학습률 변화')
    # plt.legend()
    # plt.yscale('log')  # 로그 스케일로 표시
    # plt.grid(True, linestyle='--', alpha=0.7)
    # plt.savefig('results/learning_rate.png')
    # plt.close()
      
    # 최종 모델 저장
    torch.save(model.state_dict(), 'checkpoints/final_spectrum_model.pth')
    print("\n학습 완료!")
    print(f"최적 모델 저장됨 (검증 손실: {best_val_loss:.9f})")


if __name__ == "__main__":
    # 설정 상수
    BATCH_SIZE = 2
    EPOCHS = 5
    LEARNING_RATE = 0.0001  # MobileNetV3에 맞게 학습률 조정
    NUM_CLASSES = 3  # 클래스 수 (WiFi, collision, BT)
    IMG_HEIGHT = 192
    IMG_WIDTH = 1024
    TRAIN_DIR = './spectrogram_training_data_20221006/train'
    VAL_DIR = './spectrogram_training_data_20221006/val'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # for debug
    # visualize_samples_check(img_dir='./spectrogram_training_data_20221006/train', num_samples=2)

    main()

