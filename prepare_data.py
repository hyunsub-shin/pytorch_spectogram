import os
import shutil
import random
import argparse
import cv2
import numpy as np
import re
import glob
# from glob import glob
from PIL import Image, ImageDraw
from tqdm import tqdm

def get_signal_skeleton_mask_line(img, thickness=2):
    """점들의 중심을 찾아 순서대로 선으로 연결한 골격 마스크 생성"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([10, 50, 50]) 
    upper_yellow = np.array([40, 255, 255])
    binary_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    skeleton = np.zeros_like(binary_mask)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return binary_mask, binary_mask

    # 1. 각 점(Contour)의 중심점(Center) 추출
    centers = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append([cX, cY])
        else:
            # 면적이 너무 작은 경우 대표 점 하나 추출
            centers.append(cnt[0][0].tolist())

    # 2. 점들을 x축 기준으로 정렬 (왼쪽에서 오른쪽으로 선을 잇기 위함)
    centers = sorted(centers, key=lambda x: x[0])
    pts = np.array(centers, np.int32)
    pts = pts.reshape((-1, 1, 2))

    # 3. 점들을 선으로 연결 (Skeletal Line)
    # thickness를 조절하여 선의 굵기를 결정합니다.
    cv2.polylines(skeleton, [pts], isClosed=False, color=255, thickness=thickness)

    return binary_mask, skeleton

def get_signal_skeleton_mask_polygon(img):
    """점들을 연결하여 시그널의 '모양'을 가진 마스크 생성"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([10, 50, 50]) 
    upper_yellow = np.array([40, 255, 255])
    binary_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    skeleton = np.zeros_like(binary_mask)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return binary_mask, binary_mask

    all_points = np.vstack(contours)
    hull = cv2.convexHull(all_points)
    cv2.drawContours(skeleton, [hull], -1, 255, thickness=-1) 

    return binary_mask, skeleton

def extract_class_id(filename):
    """파일명에서 마지막 숫자를 찾아 class ID로 반환 (숫자가 없으면 0)"""
    numbers = re.findall(r'\d+', filename)
    if numbers:
        # 마지막 숫자를 사용하여 class id 결정 (signal_1 -> class 0, signal_2 -> class 1)
        if (int(numbers[-1]) - 1) < 0:
            return 0
        else:
            return int(numbers[-1]) - 1 # 가장 마지막에 등장하는 숫자 반환
    else:
        print(f"경고: 파일명에서 숫자를 찾을 수 없어 클래스 0을 사용합니다: {filename}")        
        return 0

def synthesize_advanced(signal_path, bg_folder, output_root, num_gen=10, mask_type='line'):
    # 1. 파일 이름 설정 및 class ID 추출
    signal_basename = os.path.basename(signal_path)
    signal_name = os.path.splitext(signal_basename)[0]
    class_id = extract_class_id(signal_name)
    
    save_dirs = {
        "images": os.path.join(output_root, "images"),
        "labels": os.path.join(output_root, "labels"),
        "debug": os.path.join(output_root, "debug")
    }
    for p in save_dirs.values():
        os.makedirs(p, exist_ok=True)

    bg_list = glob(os.path.join(bg_folder, "*.*"))
    bg_list = [f for f in bg_list if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not bg_list:
        print("❌ 배경 폴더에 이미지가 없습니다.")
        return

    pattern_img = cv2.imread(signal_path)
    if mask_type == 'line':
        binary_mask, skeleton_mask = get_signal_skeleton_mask_line(pattern_img)
    elif mask_type == 'polygon':
        binary_mask, skeleton_mask = get_signal_skeleton_mask_polygon(pattern_img)
    else:
        print("❌ 올바른 마스크 타입이 아닙니다.")
        return

    coords = cv2.findNonZero(binary_mask)
    if coords is None:
        print("❌ 시그널 패턴을 찾을 수 없습니다.")
        return
        
    x, y, w, h = cv2.boundingRect(coords)
    crop_img = pattern_img[y:y+h, x:x+w]
    crop_binary = binary_mask[y:y+h, x:x+w]
    crop_skeleton = skeleton_mask[y:y+h, x:x+w]

    print(f"🚀 '{signal_name}' (Class ID: {class_id}) 패턴으로 {num_gen}개 생성 시작")

    for i in range(num_gen):
        bg_path = random.choice(bg_list)
        bg_img_origin = cv2.imread(bg_path)
        
        bg = bg_img_origin.copy()
        debug = bg_img_origin.copy()
        bg_h, bg_w, _ = bg.shape

        if bg_w <= w or bg_h <= h:
            continue

        rx = random.randint(0, bg_w - w)
        ry = random.randint(0, bg_h - h)

        # 이미지 합성
        roi = bg[ry:ry+h, rx:rx+w]
        bg_part = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(crop_binary))
        fg_part = cv2.bitwise_and(crop_img, crop_img, mask=crop_binary)
        bg[ry:ry+h, rx:rx+w] = cv2.add(bg_part, fg_part)

        # 라벨 생성
        contours, _ = cv2.findContours(crop_skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        yolo_labels = []
        for cnt in contours:
            abs_cnt = cnt.copy()
            abs_cnt[:, 0, 0] += rx
            abs_cnt[:, 0, 1] += ry
            
            # 디버깅 이미지에 class ID 텍스트 추가
            cv2.drawContours(debug, [abs_cnt], -1, (0, 255, 0), 2)
            cv2.putText(debug, f"ID: {class_id}", (rx, ry-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            polygon = []
            for p in abs_cnt:
                polygon.append(f"{p[0][0]/bg_w:.6f} {p[0][1]/bg_h:.6f}")
            # 첫 번째 인자에 추출한 class_id 적용
            yolo_labels.append(f"{class_id} {' '.join(polygon)}")

        save_name = f"{signal_name}_{i:04d}"
        cv2.imwrite(os.path.join(save_dirs["images"], f"{save_name}.png"), bg)
        cv2.imwrite(os.path.join(save_dirs["debug"], f"{save_name}_debug.png"), debug)
        with open(os.path.join(save_dirs["labels"], f"{save_name}.txt"), 'w') as f:
            f.write('\n'.join(yolo_labels))

    print(f"✅ 완료: {output_root}")
    
def delete_files_with_suffix(directory, suffix):
    """
    특정 접미사를 포함하는 파일 삭제
    
    Args:
        directory: 대상 디렉토리 경로
        suffix: 삭제할 파일의 접미사
    """
    # 디렉토리 내의 모든 파일을 순회
    for filename in tqdm(os.listdir(directory)):
        # 파일명이 접미사로 끝나는지 확인
        if filename.endswith(suffix):
            file_path = os.path.join(directory, filename)
            try:
                # 파일 삭제
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

def create_directory_structure(base_dir):
    """
    학습/검증/테스트를 위한 디렉토리 구조 생성
    
    기본 구조:
        base_dir/train
        base_dir/val
        base_dir/test
    
    (YOLO 전용 구조는 `split_dataset`에서 별도로 생성)
    
    Args:
        base_dir: 기본 디렉토리 경로
    """
    os.makedirs(base_dir, exist_ok=True)
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(base_dir, split), exist_ok=True)
    
    print(f"디렉토리 구조 생성 완료: {base_dir}")

def split_dataset(images_dir, labels_dir, masks_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_seed=42):
    """
    데이터셋을 학습/검증/테스트로 분할 (YOLO 데이터셋 구조로 저장)
    
    Args:
        images_dir: 이미지 디렉토리 경로
        labels_dir: 라벨 디렉토리 경로 (YOLO 형식 txt 파일)
        masks_dir: 마스크 디렉토리 경로 (선택적)
        output_dir: 출력 디렉토리 경로 (예: ./datasets)
            - images/train, images/val, images/test
            - labels/train, labels/val, labels/test
            - (선택) masks/train, masks/val, masks/test
        train_ratio: 학습 데이터 비율
        val_ratio: 검증 데이터 비율
        test_ratio: 테스트 데이터 비율
        random_seed: 랜덤 시드
    """
    # YOLO 형식 디렉토리 구조 생성
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)
        if masks_dir:
            os.makedirs(os.path.join(output_dir, 'masks', split), exist_ok=True)
    
    # 이미지 파일 목록 가져오기
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(images_dir) if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    # 랜덤 시드 설정
    random.seed(random_seed)
    random.shuffle(image_files)
    
    # 분할 인덱스 계산
    total = len(image_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    # 분할 데이터 준비
    splits = {
        'train': image_files[:train_end],
        'val': image_files[train_end:val_end],
        'test': image_files[val_end:]
    }
    
    # 각 분할에 대해 파일 복사 (YOLO 구조: images/…, labels/…, masks/…)
    for split, files in splits.items():
        print(f"{split} 데이터 복사 중: {len(files)} 파일")
        for file in tqdm(files):
            # 이미지 파일 복사
            src_image = os.path.join(images_dir, file)
            dst_image = os.path.join(output_dir, 'images', split, file)
            shutil.copy2(src_image, dst_image)
            
            # 라벨 파일 복사 (있는 경우)
            base_name = os.path.splitext(file)[0]
            src_label = os.path.join(labels_dir, f"{base_name}.txt")
            dst_label = os.path.join(output_dir, 'labels', split, f"{base_name}.txt")
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)
            
            # 마스크 파일 복사 (있는 경우)
            if masks_dir:
                src_mask = os.path.join(masks_dir, f"{base_name}_mask.png")
                dst_mask = os.path.join(output_dir, 'masks', split, f"{base_name}_mask.png")
                if os.path.exists(src_mask):
                    shutil.copy2(src_mask, dst_mask)
    
    print("데이터셋 분할 완료!")
    print(f"학습: {len(splits['train'])} 파일")
    print(f"검증: {len(splits['val'])} 파일")
    print(f"테스트: {len(splits['test'])} 파일")

def split_sliced_dataset(image_dir, label_dir, output_base_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_seed=42):
    """
    image_dir: 슬라이싱된 원본 이미지 폴더
    label_dir: 슬라이싱된 원본 라벨 폴더
    output_base_dir: 최종 데이터셋 루트
            - images/train, images/val, images/test
            - labels/train, labels/val, labels/test
    """
    # 1. 이미지 파일 목록 가져오기
    all_images = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # 2. 원본 이름 단위로 그룹화 (파일명 규칙: {name}_{x}_{y}.png)
    original_groups = {}
    for f in all_images:
        parts = f.rsplit('_', 2)
        orig_name = parts[0]
        if orig_name not in original_groups:
            original_groups[orig_name] = []
        original_groups[orig_name].append(f)

    # 3. 원본 이름 리스트 셔플 (랜덤성 부여)
    orig_names = list(original_groups.keys())
    random.seed(random_seed)
    random.shuffle(orig_names)

    # 4. 분할 지점 계산
    total_count = len(orig_names)
    train_end = int(total_count * train_ratio)
    val_end = train_end + int(total_count * val_ratio)

    split_map = {
        'train': orig_names[:train_end],
        'val': orig_names[train_end:val_end],
        'test': orig_names[val_end:]
    }

    print(f"--- [데이터 분할 시작: 사용자 지정 구조] ---")
    print(f"전체 원본 이미지 수: {total_count}")

    # 5. 파일 복사 및 폴더 정리
    for split_name, orig_list in split_map.items():
        # 사용자 요청 구조: output_base_dir/images/train... 및 output_base_dir/labels/train...
        target_img_dir = os.path.join(output_base_dir, 'images', split_name)
        target_lbl_dir = os.path.join(output_base_dir, 'labels', split_name)
        
        os.makedirs(target_img_dir, exist_ok=True)
        os.makedirs(target_lbl_dir, exist_ok=True)

        for orig in tqdm(orig_list, desc=f"정리 중: {split_name}"):
            for img_file in original_groups[orig]:
                # 1. 이미지 복사 (src -> images/{train,val,test})
                shutil.copy(
                    os.path.join(image_dir, img_file), 
                    os.path.join(target_img_dir, img_file)
                )
                
                # 2. 대응하는 라벨 복사 (src -> labels/{train,val,test})
                lbl_file = os.path.splitext(img_file)[0] + '.txt'
                src_lbl_path = os.path.join(label_dir, lbl_file)
                
                if os.path.exists(src_lbl_path):
                    shutil.copy(src_lbl_path, os.path.join(target_lbl_dir, lbl_file))

    print(f"\n✅ 분할 완료!")
    print(f"📂 구조 확인:")
    print(f"   - {output_base_dir}/images/{{train, val, test}}")
    print(f"   - {output_base_dir}/labels/{{train, val, test}}")

def create_mask_from_bbox(image_path, label_path, output_path, dilation_size=5):
    """
    바운딩 박스로부터 마스크 이미지 생성
    
    Args:
        image_path: 원본 이미지 경로
        label_path: 라벨 파일 경로 (YOLO 형식 txt)
        output_path: 출력 마스크 이미지 경로
        dilation_size: 바운딩 박스를 팽창시킬 크기 (픽셀)
    """
    # 이미지 로드
    image = Image.open(image_path)
    width, height = image.size
    
    # 마스크 생성 (검은색 배경)
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    # 라벨 파일 읽기
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                values = line.strip().split()
                if len(values) == 5:
                    # YOLO 형식: class_id x_center y_center width height
                    class_id = int(values[0])
                    x_center = float(values[1]) * width
                    y_center = float(values[2]) * height
                    box_width = float(values[3]) * width
                    box_height = float(values[4]) * height
                    
                    # 바운딩 박스 좌표 계산
                    x1 = max(0, int(x_center - box_width / 2))
                    y1 = max(0, int(y_center - box_height / 2))
                    x2 = min(width - 1, int(x_center + box_width / 2))
                    y2 = min(height - 1, int(y_center + box_height / 2))
                    
                    # 바운딩 박스 내부를 흰색으로 채우기
                    draw.rectangle([x1, y1, x2, y2], fill=255)
    
    # 팽창 적용 (오브젝트 사이의 배경을 더 잘 분리하기 위함)
    if dilation_size > 0:
        # PIL 이미지를 OpenCV 형식으로 변환
        mask_np = np.array(mask)
        kernel = np.ones((dilation_size, dilation_size), np.uint8)
        dilated_mask = cv2.dilate(mask_np, kernel, iterations=1)
        
        # 다시 PIL 이미지로 변환
        mask = Image.fromarray(dilated_mask)
    
    # 마스크 저장
    mask.save(output_path)
    return mask

def create_masks_for_dataset(dataset_dir, output_dir=None, dilation_size=5):
    """
    데이터셋의 모든 이미지에 대해 마스크 생성
    
    Args:
        dataset_dir: 데이터셋 디렉토리 (이미지와 라벨 파일 포함)
        output_dir: 마스크 출력 디렉토리 (None이면 dataset_dir과 동일)
        dilation_size: 바운딩 박스를 팽창시킬 크기 (픽셀)
    """
    if output_dir is None:
        output_dir = dataset_dir
    
    # 이미지 파일 목록 가져오기
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(dataset_dir) 
                   if any(f.lower().endswith(ext) for ext in image_extensions) 
                   and '_masked' not in f]
    
    print(f"마스크 생성 중: {len(image_files)} 파일")
    for file in tqdm(image_files):
        # 파일 경로
        image_path = os.path.join(dataset_dir, file)
        base_name = os.path.splitext(file)[0]
        label_path = os.path.join(dataset_dir, f"{base_name}.txt")
        mask_output_path = os.path.join(output_dir, f"{base_name}_masked.png")
        
        # 마스크 생성
        create_mask_from_bbox(image_path, label_path, mask_output_path, dilation_size)
    
    print(f"마스크 생성 완료: {len(image_files)} 파일")

def augment_dataset(dataset_dir, num_augmentations=5, random_seed=42):
    """
    데이터셋 증강: 회전, 색상 변화, 노이즈 추가 등
    
    Args:
        dataset_dir: 데이터셋 디렉토리
        num_augmentations: 각 이미지당 생성할 증강 이미지 수
        random_seed: 랜덤 시드
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # 이미지 파일 목록 가져오기
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(dataset_dir) 
                   if any(f.lower().endswith(ext) for ext in image_extensions) 
                   and '_mask' not in f 
                   and '_aug' not in f]
    
    print(f"데이터 증강 중: {len(image_files)} 파일 x {num_augmentations} 증강 = {len(image_files) * num_augmentations} 파일")
    
    for file in tqdm(image_files):
        # 파일 경로
        image_path = os.path.join(dataset_dir, file)
        base_name = os.path.splitext(file)[0]
        ext = os.path.splitext(file)[1]
        label_path = os.path.join(dataset_dir, f"{base_name}.txt")
        mask_path = os.path.join(dataset_dir, f"{base_name}_mask.png")
        
        # 이미지 로드
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        
        # 라벨 로드
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    labels.append(line.strip())
        
        # 마스크 로드 (있는 경우)
        mask = None
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 각 증강에 대해
        for i in range(num_augmentations):
            # 랜덤 증강 적용
            aug_image = image.copy()
            aug_mask = mask.copy() if mask is not None else None
            
            # 1. 회전 (작은 각도)
            angle = random.uniform(-15, 15)
            M = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
            aug_image = cv2.warpAffine(aug_image, M, (width, height))
            if aug_mask is not None:
                aug_mask = cv2.warpAffine(aug_mask, M, (width, height))
            
            # 2. 밝기 조정
            brightness = random.uniform(0.8, 1.2)
            aug_image = cv2.convertScaleAbs(aug_image, alpha=brightness, beta=0)
            
            # 3. 대비 조정
            contrast = random.uniform(0.8, 1.2)
            aug_image = cv2.convertScaleAbs(aug_image, alpha=contrast, beta=0)
            
            # 4. 노이즈 추가
            if random.random() < 0.5:
                noise = np.random.normal(0, 5, aug_image.shape).astype(np.uint8)
                aug_image = cv2.add(aug_image, noise)
            
            # 5. 좌우 반전
            if random.random() < 0.5:
                aug_image = cv2.flip(aug_image, 1)
                if aug_mask is not None:
                    aug_mask = cv2.flip(aug_mask, 1)
                
                # 라벨도 반전해야 함
                for j in range(len(labels)):
                    parts = labels[j].split()
                    if len(parts) == 5:
                        class_id, x_center, y_center, w, h = parts
                        # x 좌표 반전
                        x_center = str(1.0 - float(x_center))
                        labels[j] = f"{class_id} {x_center} {y_center} {w} {h}"
            
            # 증강된 이미지 저장
            aug_image_path = os.path.join(dataset_dir, f"{base_name}_aug{i+1}{ext}")
            cv2.imwrite(aug_image_path, aug_image)
            
            # 증강된 마스크 저장 (있는 경우)
            if aug_mask is not None:
                aug_mask_path = os.path.join(dataset_dir, f"{base_name}_aug{i+1}_mask.png")
                cv2.imwrite(aug_mask_path, aug_mask)
            
            # 증강된 라벨 저장
            if labels:
                aug_label_path = os.path.join(dataset_dir, f"{base_name}_aug{i+1}.txt")
                with open(aug_label_path, 'w') as f:
                    for label in labels:
                        f.write(f"{label}\n")
    
    print("데이터 증강 완료!")

def preview_dataset(dataset_dir, num_samples=5, random_seed=42):
    """
    데이터셋 미리보기 (이미지, 바운딩 박스, 마스크 시각화)
    
    Args:
        dataset_dir: 데이터셋 디렉토리
        num_samples: 미리볼 샘플 수
        random_seed: 랜덤 시드
    """
    import matplotlib.pyplot as plt
    
    random.seed(random_seed)
    
    # 이미지 파일 목록 가져오기
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(dataset_dir) 
                   if any(f.lower().endswith(ext) for ext in image_extensions) 
                   and '_mask' not in f]
    
    # 랜덤 샘플링
    if len(image_files) > num_samples:
        image_files = random.sample(image_files, num_samples)
    
    for file in image_files:
        # 파일 경로
        image_path = os.path.join(dataset_dir, file)
        base_name = os.path.splitext(file)[0]
        label_path = os.path.join(dataset_dir, f"{base_name}.txt")
        mask_path = os.path.join(dataset_dir, f"{base_name}_mask.png")
        
        # 이미지 로드
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        # 마스크 로드 (있는 경우)
        mask = None
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 서브플롯 설정
        fig, axes = plt.subplots(1, 3 if mask is not None else 2, figsize=(15, 5))
        
        # 원본 이미지 표시
        axes[0].imshow(image)
        axes[0].set_title(f"원본 이미지: {file}")
        axes[0].axis('off')
        
        # 바운딩 박스가 포함된 이미지 표시
        axes[1].imshow(image)
        axes[1].set_title("바운딩 박스")
        axes[1].axis('off')
        
        # 라벨 파일 읽기 및 바운딩 박스 그리기
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    values = line.strip().split()
                    if len(values) == 5:
                        # YOLO 형식: class_id x_center y_center width height
                        class_id = int(values[0])
                        x_center = float(values[1]) * width
                        y_center = float(values[2]) * height
                        box_width = float(values[3]) * width
                        box_height = float(values[4]) * height
                        
                        # 바운딩 박스 좌표 계산
                        x1 = max(0, int(x_center - box_width / 2))
                        y1 = max(0, int(y_center - box_height / 2))
                        x2 = min(width - 1, int(x_center + box_width / 2))
                        y2 = min(height - 1, int(y_center + box_height / 2))
                        
                        # 클래스별 색상 설정
                        color = (1, 0, 0) if class_id == 0 else (0, 1, 0)  # 클래스 0은 빨간색, 클래스 1은 녹색
                        
                        # 바운딩 박스 그리기
                        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                          edgecolor=color, facecolor='none', linewidth=2)
                        axes[1].add_patch(rect)
                        
                        # 클래스 레이블 표시
                        axes[1].text(x1, y1 - 5, f"Class {class_id}", 
                                  color='white', fontsize=8, 
                                  bbox=dict(facecolor=color, alpha=0.7))
        
        # 마스크 표시 (있는 경우)
        if mask is not None:
            axes[2].imshow(mask, cmap='gray')
            axes[2].set_title("마스크")
            axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()

def create_synthetic_drone_images(drone_model_path, background_dir, output_dir, num_images=100):
    """드론 모델을 다양한 배경에 합성하여 이미지 생성"""
    
    # 드론 시그널 파일명에서 클래스 ID 추출 및 파일명 기반 이름 생성
    # 예: signal_1.png -> 클래스 0, signal_2.png -> 클래스 1
    drone_filename = os.path.basename(drone_model_path)
    # 확장자 제거하여 기본 파일명 추출
    drone_base_name = os.path.splitext(drone_filename)[0]
    
    # 파일명에서 숫자 추출
    numbers = re.findall(r'\d+', drone_filename)
    if numbers:
        # 마지막 숫자를 사용하여 클래스 ID 결정 (signal_1 -> 클래스 0, signal_2 -> 클래스 1)
        class_id = int(numbers[-1]) - 1
        if class_id < 0:
            class_id = 0
    else:
        # 숫자가 없으면 기본값 0 사용
        class_id = 0
        print(f"경고: 파일명에서 숫자를 찾을 수 없어 클래스 0을 사용합니다: {drone_filename}")
    
    print(f"드론 시그널: {drone_filename} -> 클래스 ID: {class_id}, 기본 파일명: {drone_base_name}")
    
    # 드론 모델 이미지 로드 (알파 채널 포함)
    drone = cv2.imread(drone_model_path, cv2.IMREAD_UNCHANGED)
    
    # 드론 이미지 디버깅
    if drone is None:
        print(f"오류: 드론 이미지를 로드할 수 없습니다: {drone_model_path}")
        return
    
    print(f"드론 이미지 정보: 크기={drone.shape}, 타입={drone.dtype}")
    
    # 드론 이미지에 알파 채널이 없으면 추가
    if len(drone.shape) == 2:  # 그레이스케일 이미지
        drone = cv2.cvtColor(drone, cv2.COLOR_GRAY2BGRA)
        drone[:, :, 3] = 255  # 완전 불투명 설정
    elif drone.shape[2] == 3:  # BGR 이미지, 알파 채널 없음
        b, g, r = cv2.split(drone)
        alpha = np.ones(b.shape, dtype=b.dtype) * 255
        drone = cv2.merge((b, g, r, alpha))
    
    # 배경 이미지 목록
    backgrounds = [os.path.join(background_dir, f) for f in os.listdir(background_dir)
                  if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    if not backgrounds:
        print(f"오류: 배경 이미지를 찾을 수 없습니다: {background_dir}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_images):
        # 랜덤 배경 선택
        bg_path = random.choice(backgrounds)
        background = cv2.imread(bg_path)
        
        if background is None:
            print(f"오류: 배경 이미지를 로드할 수 없습니다: {bg_path}")
            continue
        
        # 배경 크기 조정
        bg_h, bg_w = background.shape[:2]
        print(f"배경 크기: {bg_w}x{bg_h}")
        
        # 드론 크기 및 위치 랜덤화
        # 원본 드론 크기
        drone_orig_h, drone_orig_w = drone.shape[:2]
        print(f"드론 원본 크기: {drone_orig_w}x{drone_orig_h}")
        
        # 드론이 배경보다 크면 배경에 맞게 크기 조정 (이 경우만 크기 조정)
        if drone_orig_w > bg_w or drone_orig_h > bg_h:
            max_width = int(bg_w)  # 배경의 90%로 제한
            max_height = int(bg_h)  # 배경의 90%로 제한
            
            # 비율 유지하며 크기 조정
            width_ratio = max_width / drone_orig_w
            height_ratio = max_height / drone_orig_h
            scale_factor = min(width_ratio, height_ratio)
            
            print(f"드론 이미지가 배경보다 큽니다. 스케일 팩터 {scale_factor}로 조정합니다.")
            scale = scale_factor
        else:
            # 원본 크기 그대로 사용
            scale = 1.0  # 원본 크기 유지
            
        print(f"적용된 스케일: {scale}, 드론 원본 크기: {drone_orig_w}x{drone_orig_h}")
        drone_resized = cv2.resize(drone, (0, 0), fx=scale, fy=scale)
        
        # 드론 회전 적용 제외
        drone_h, drone_w = drone_resized.shape[:2]
        print(f"크기 조정 후 드론 크기: {drone_w}x{drone_h}, 배경 크기: {bg_w}x{bg_h}")
        
        # 회전 없이 그대로 사용
        drone_rotated = drone_resized
        
        # 드론 위치 선택 (화면 중앙에 가깝게)
        # 배경 이미지에서 드론이 들어갈 수 있는 최대 범위 계산
        max_x = max(1, bg_w - drone_w)
        max_y = max(1, bg_h - drone_h)
        
        # 가능한 범위 내에서 중앙 근처에 배치
        x_left = max(0, min(bg_w//4, max_x//4))
        x_right = max(x_left + 1, min(bg_w*3//4, max_x))
        y_top = max(0, min(bg_h//4, max_y//4))
        y_bottom = max(y_top + 1, min(bg_h*3//4, max_y))
        
        x_pos = random.randint(x_left, x_right)
        y_pos = random.randint(y_top, y_bottom)
        # y_pos = (0)
        
        print(f"드론 위치: ({x_pos}, {(y_pos)})")
        
        # 합성할 배경 영역 준비
        roi = background[y_pos:y_pos+drone_h, x_pos:x_pos+drone_w]
        
        # 알파 블렌딩으로 이미지 합성
        try:
            if drone_rotated.shape[2] == 4:  # 알파 채널 존재
                # 알파 마스크 추출
                alpha_mask = drone_rotated[:, :, 3] / 255.0
                alpha_mask_3d = np.stack([alpha_mask] * 3, axis=2)
                
                # 알파 블렌딩
                foreground = drone_rotated[:, :, :3]
                blended_img = foreground * alpha_mask_3d + roi * (1 - alpha_mask_3d)
                
                # 합성 결과를 배경에 적용
                background[y_pos:y_pos+drone_h, x_pos:x_pos+drone_w] = blended_img
            else:
                # 알파 채널이 없는 경우 그냥 복사
                background[y_pos:y_pos+drone_h, x_pos:x_pos+drone_w] = drone_rotated[:, :, :3]
        except Exception as e:
            print(f"이미지 합성 중 오류 발생: {e}")
            print(f"드론 크기: {drone_rotated.shape}, ROI 크기: {roi.shape}")
            continue
        
        # YOLO 형식의 바운딩 박스 라벨 생성
        drone_center_x = (x_pos + drone_w/2) / bg_w
        drone_center_y = (y_pos + drone_h/2) / bg_h
        drone_width = drone_w / bg_w
        drone_height = drone_h / bg_h
        
        # 드론 파일명 기반으로 저장 파일명 생성
        output_filename = f"{drone_base_name}_{i:04d}.png"
        label_filename = f"{drone_base_name}_{i:04d}.txt"
        
        # 이미지 저장
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, background)
        
        # 라벨 저장 (YOLO 형식) - 드론 시그널에 따라 클래스 ID 사용
        label_path = os.path.join(output_dir, label_filename)
        with open(label_path, 'w') as f:
            f.write(f"{class_id} {drone_center_x} {drone_center_y} {drone_width} {drone_height}\n")
    
    print(f"{num_images}개의 합성 이미지가 {output_dir}에 생성되었습니다.")

def batch_slice_yolo_polygon(input_img_dir, input_label_dir, output_dir, tile_size=1024, overlap=0.1):
    # 1. 경로 설정
    out_img_path = os.path.join(output_dir, 'images')
    out_label_path = os.path.join(output_dir, 'labels')
    os.makedirs(out_img_path, exist_ok=True)
    os.makedirs(out_label_path, exist_ok=True)

    img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    img_files = []
    for ext in img_extensions:
        img_files.extend(glob.glob(os.path.join(input_img_dir, ext)))

    print(f"🚀 총 {len(img_files)}개 파일 슬라이싱 시작 (Polygon 지원 모드)")

    for img_path in tqdm(img_files):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(input_label_dir, f"{img_name}.txt")

        if not os.path.exists(label_path):
            continue

        image = cv2.imread(img_path)
        if image is None: continue
        h, w, _ = image.shape
        
        with open(label_path, 'r') as f:
            lines = f.readlines()

        step = int(tile_size * (1 - overlap))

        for y in range(0, h, step):
            for x in range(0, w, step):
                # 타일 경계 계산
                x_end = min(x + tile_size, w)
                y_end = min(y + tile_size, h)
                x_start = max(0, x_end - tile_size)
                y_start = max(0, y_end - tile_size)

                tile_labels = []

                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5: continue # 최소 class + 4개좌표 필요
                    
                    class_id = parts[0]
                    coords = list(map(float, parts[1:]))
                    
                    # 박스 라벨 형식인지 확인 (5개 토큰: cls xc yc w h)
                    if len(parts) == 5:
                        # 박스 라벨 처리: cls xc yc w h -> 폴리곤 좌표로 변환
                        xc_n, yc_n, w_n, h_n = coords
                        
                        # 픽셀 좌표로 복원
                        abs_xc = xc_n * w
                        abs_yc = yc_n * h
                        abs_w = w_n * w
                        abs_h = h_n * h
                        
                        # 박스의 4개 모서리 좌표 계산
                        x1 = abs_xc - abs_w / 2
                        y1 = abs_yc - abs_h / 2
                        x2 = abs_xc + abs_w / 2
                        y2 = abs_yc - abs_h / 2
                        x3 = abs_xc + abs_w / 2
                        y3 = abs_yc + abs_h / 2
                        x4 = abs_xc - abs_w / 2
                        y4 = abs_yc + abs_h / 2
                        
                        px_pts = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
                    else:
                        # 폴리곤 라벨 형식: cls x1 y1 x2 y2 x3 y3 ...
                        # 픽셀 좌표로 복원
                        px_pts = []
                        for i in range(0, len(coords), 2):
                            px_pts.append((coords[i] * w, coords[i+1] * h))
                    
                    # --- [핵심] 타일 내부에 포함된 점들만 필터링 및 변환 ---
                    new_poly = []
                    for pt_x, pt_y in px_pts:
                        # 1. 점이 타일 밖에 있더라도 타일 경계선으로 강제 고정 (Clamping)
                        # 이를 통해 잘린 객체도 타일 경계에 붙은 올바른 폴리곤 형태를 유지함
                        cx = max(x_start, min(pt_x, x_end))
                        cy = max(y_start, min(pt_y, y_end))
                        
                        # 2. 타일 상대 좌표로 변환 및 정규화
                        # 타일 내에서의 0.0 ~ 1.0 사이 값으로 변환함
                        nx = (cx - x_start) / (x_end - x_start) 
                        ny = (cy - y_start) / (y_end - y_start)
                        new_poly.append(f"{nx:.6f} {ny:.6f}")

                    # 3. 중복된 점 제거 및 유효성 검사
                    # 폴리곤은 최소 3개의 서로 다른 점이 있어야 유효한 라벨이 됨
                    if len(set(new_poly)) >= 3:
                        tile_labels.append(f"{class_id} {' '.join(new_poly)}")

                # 객체가 있는 타일만 저장
                if tile_labels:
                    tile_img = image[y_start:y_end, x_start:x_end]
                    save_name = f"{img_name}_{x_start}_{y_start}"
                    
                    cv2.imwrite(os.path.join(out_img_path, f"{save_name}.jpg"), tile_img)
                    with open(os.path.join(out_label_path, f"{save_name}.txt"), 'w') as f_out:
                        f_out.write("\n".join(tile_labels))

def main():
    parser = argparse.ArgumentParser(description="객체 인식 데이터셋 준비 도구")
    subparsers = parser.add_subparsers(dest='command', help='명령어')
    
    # 디렉토리 구조 생성
    create_parser = subparsers.add_parser('create', help='디렉토리 구조 생성')
    create_parser.add_argument('--dir', type=str, required=True, help='기본 디렉토리 경로')
    
    # 데이터셋 분할
    split_parser = subparsers.add_parser('split', help='데이터셋을 학습/검증/테스트로 분할')
    split_parser.add_argument('--images', type=str, required=True, help='이미지 디렉토리 경로')
    split_parser.add_argument('--labels', type=str, required=True, help='라벨 디렉토리 경로')
    split_parser.add_argument('--masks', type=str, help='마스크 디렉토리 경로 (선택적)')
    split_parser.add_argument('--output', type=str, required=True, help='출력 디렉토리 경로')
    split_parser.add_argument('--train', type=float, default=0.7, help='학습 데이터 비율')
    split_parser.add_argument('--val', type=float, default=0.2, help='검증 데이터 비율')
    split_parser.add_argument('--test', type=float, default=0.1, help='테스트 데이터 비율')
    split_parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')

    # 슬라이싱 데이터셋 분할
    split_sliced_parser = subparsers.add_parser('split_sliced', help='데이터셋을 학습/검증/테스트로 분할')
    split_sliced_parser.add_argument('--images', type=str, required=True, help='이미지 디렉토리 경로')
    split_sliced_parser.add_argument('--labels', type=str, required=True, help='라벨 디렉토리 경로')
    split_sliced_parser.add_argument('--output', type=str, required=True, help='출력 디렉토리 경로')
    split_sliced_parser.add_argument('--train', type=float, default=0.7, help='학습 데이터 비율')
    split_sliced_parser.add_argument('--val', type=float, default=0.2, help='검증 데이터 비율')
    split_sliced_parser.add_argument('--test', type=float, default=0.1, help='테스트 데이터 비율')
    split_sliced_parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')
    
    # 마스크 생성
    mask_parser = subparsers.add_parser('mask', help='바운딩 박스로부터 마스크 생성')
    mask_parser.add_argument('--dir', type=str, required=True, help='데이터셋 디렉토리 (이미지와 라벨 포함)')
    mask_parser.add_argument('--output', type=str, help='마스크 출력 디렉토리 (선택적)')
    mask_parser.add_argument('--dilation', type=int, default=5, help='바운딩 박스 팽창 크기 (픽셀)')
    
    # 데이터 증강
    augment_parser = subparsers.add_parser('augment', help='데이터셋 증강')
    augment_parser.add_argument('--dir', type=str, required=True, help='데이터셋 디렉토리')
    augment_parser.add_argument('--num', type=int, default=5, help='각 이미지당 생성할 증강 이미지 수')
    augment_parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')
    
    # 데이터셋 미리보기
    preview_parser = subparsers.add_parser('preview', help='데이터셋 미리보기')
    preview_parser.add_argument('--dir', type=str, required=True, help='데이터셋 디렉토리')
    preview_parser.add_argument('--num', type=int, default=5, help='미리볼 샘플 수')
    preview_parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')
    
    # 데이터셋 만들기
    synthetic_parser = subparsers.add_parser('synthetic', help='데이터셋 만들기')
    synthetic_parser.add_argument('--drone', type=str, required=True, help='드론 이미지')
    synthetic_parser.add_argument('--back', type=str, required=True, help='백그라운드 이미지 디렉토리')
    synthetic_parser.add_argument('--output', type=str, required=True, help='출력 디렉토리')
    synthetic_parser.add_argument('--num', type=int, default=5, help='이미지 생성 개수')
    
    # 데이터셋 만들기 (스케일럿 마스크 적용)
    synthetic_skeletal_parser = subparsers.add_parser('synthetic_skeletal', help='데이터셋 만들기 (스케일럿 마스크 적용)')
    synthetic_skeletal_parser.add_argument('--drone', type=str, required=True, help='드론 이미지')
    synthetic_skeletal_parser.add_argument('--back_dir', type=str, required=True, help='백그라운드 이미지 디렉토리')
    synthetic_skeletal_parser.add_argument('--output_dir', type=str, required=True, help='출력 디렉토리')
    synthetic_skeletal_parser.add_argument('--num', type=int, default=5, help='이미지 생성 개수')
    synthetic_skeletal_parser.add_argument('--mask_type', type=str, default='line', help='마스크 타입 (line, polygon)')

    # 데이터셋셋 슬라이스
    slicing_image_parser = subparsers.add_parser('img_slice', help='데이터셋 슬라이스 만들기(큰이미지를 작게 쪼개기기)')
    slicing_image_parser.add_argument('--img_dir', type=str, required=True, help='이미지 디렉토리')
    slicing_image_parser.add_argument('--label_dir', type=str, required=True, help='라벨벨 디렉토리')
    slicing_image_parser.add_argument('--output_dir', type=str, required=True, help='출력 디렉토리')
    slicing_image_parser.add_argument('--size', type=int, default=128, help='슬라이스 이미지 크기')
    slicing_image_parser.add_argument('--overlap', type=float, default=0.1, help='오버랩 크기기')
    
    # 파일 삭제
    delete_parser = subparsers.add_parser('delete', help='특정 접미사를 포함하는 파일 삭제')
    delete_parser.add_argument('--dir', type=str, required=True, help='대상 디렉토리 경로')
    delete_parser.add_argument('--suffix', type=str, required=True, help='삭제할 파일의 접미사')
    
    args = parser.parse_args()
    
    if args.command == 'create':
        create_directory_structure(args.dir)
    elif args.command == 'split':
        split_dataset(args.images, args.labels, args.masks, args.output, 
                     args.train, args.val, args.test, args.seed)
    elif args.command == 'split_sliced':
        split_sliced_dataset(args.images, args.labels, args.output, 
                     args.train, args.val, args.test, args.seed)
    elif args.command == 'mask':
        create_masks_for_dataset(args.dir, args.output, args.dilation)
    elif args.command == 'augment':
        augment_dataset(args.dir, args.num, args.seed)
    elif args.command == 'preview':
        preview_dataset(args.dir, args.num, args.seed)
    elif args.command == 'synthetic':
        create_synthetic_drone_images(args.drone, args.back, args.output, args.num)
    elif args.command == 'synthetic_skeletal':
        synthesize_advanced(args.drone, args.back_dir, args.output_dir, args.num, args.mask_type)
    elif args.command == 'slice_image':
        batch_slice_yolo_polygon(args.img_dir, args.label_dir, args.output_dir, args.size, args.overlap)
    elif args.command == 'delete':
        delete_files_with_suffix(args.dir, args.suffix)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 

########################################################
## command example
########################################################
img_dir = './datasets/synthetic/images'
label_dir = './datasets/synthetic/labels'
drone_path = './datasets/drone_data/signal/autelevo_05_sig_2.png'    # 사용할 시그널 이미지 파일
background_dir = './datasets/drone_data/background'     # 배경 이미지들이 들어있는 폴더 경로
synthetic_output_dir = './datasets/synthetic' # 합성 이미지 저장될 경로
sliced_out_dir = './datasets/synthetic/sliced_data'
final_output_dir = './datasets' # 최종 분할 데이터셋 저장될 경로

# # 드론 이미지 합성 (박스 마스크 적용)
# # python prepare_data.py synthetic --drone ./datasets/drone_data/signal/autelevo_01_sig_2.png --back ./datasets/drone_data/background --output ./datasets/synthetic --num 100
# create_synthetic_drone_images(drone_path, background_dir, synthetic_output_dir, num_images=50)

# # 드론 이미지 합성 (스케일럿 마스크 적용)
# # python prepare_data.py synthetic_skeletal --drone ./datasets/drone_data/signal/signal_4.png --back_dir ./datasets/drone_data/background --output_dir ./datasets/synthetic --num 5 --mask_type line
# synthesize_advanced(drone_path, background_dir, synthetic_output_dir, num_gen=50, mask_type='line') # mask_type='polygon')

# # 합성 이미지 슬라이싱
# python prepare_data.py slice_image --img_dir ./datasets/drone_data/signal --label_dir ./datasets/drone_data/background --output_dir ./datasets/synthetic --size 2560 --overlap 0.3
batch_slice_yolo_polygon(img_dir, label_dir, sliced_out_dir, tile_size=2560, overlap=0.3)

# # 원본 이미지 분할 (train, val, test)   
# # python prepare_data.py split --images ./datasets/synthetic --labels ./datasets/synthetic --output ./datasets --train 0.7 --val 0.2 --test 0.1
# split_dataset(img_dir, label_dir, None, final_output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)

# # 슬라이싱 이미지 묶음 분할 (train, val, test)
# # python prepare_data.py split_sliced --images ./datasets/synthetic/images --labels ./datasets/synthetic/labels --output ./datasets --train 0.7 --val 0.2 --test 0.1
split_sliced_dataset(img_dir, label_dir, final_output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)