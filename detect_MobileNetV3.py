import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from models.model_MobileNetV3 import SpectrumModel
from dataset import Compose, Resize, ToTensor, Normalize
import argparse
import sys
import platform  # 폰트관련 운영체제 확인

# plot용 한글 폰트 설정
if platform.system() == 'Windows':
    font_name = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    font_name = 'AppleGothic'
else:
    font_name = 'NanumGothic'

plt.rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False

# 설정
IMG_WIDTH = 1024  # 원본 이미지와 같은 비율 유지
IMG_HEIGHT = 192
NUM_CLASSES = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["WiFi", "collision", "BT"]  # 클래스 이름 지정
# 객체 탐지 모델이 예측한 바운딩 박스의 신뢰도 임계값
CONF_THRESHOLD = 0.8  # 신뢰도 임계값을 더 낮춤 (0.05 -> 0.01)
# 하나의 객체에 대해 여러 개의 중복된 바운딩 박스가 생성되는 것을 방지
NMS_THRESHOLD = 0.35   # Non-Maximum Suppression 임계값을 약간 낮춤

# 클래스별 색상 정의 (R, G, B) - 전역 변수로 설정
CLASS_COLORS = {
    0: (0, 255, 0),     # WiFi: 녹색
    1: (255, 0, 0),     # collision: 빨간색
    2: (0, 0, 255)      # BT: 파란색
}

def load_model(model_path, device):
    """모델 로드 함수
    
    Args:
        model_path: 모델 파일 경로
        device: 장치 (CPU/GPU)
    
    Returns:
        loaded_model: 로드된 모델
    """
    # 모델 초기화 (MobileNetV3 기반)
    model = SpectrumModel(num_classes=3, pretrained=False).to(device)
    
    # 모델 가중치 로드
    try:
        # 체크포인트 로드 시도 (train.py에서 저장한 형식)
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        
        # 체크포인트에 'model_state_dict'가 있는지 확인 (체크포인트 형식)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"체크포인트에서 모델을 성공적으로 로드했습니다. (에폭: {checkpoint.get('epoch', 'N/A')})")
        else:
            # 직접 state_dict가 저장된 형식
            model.load_state_dict(checkpoint)
            print(f"모델을 {model_path}에서 성공적으로 로드했습니다.")
        
        print(f"모델을 {model_path}에서 성공적으로 로드했습니다.")
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        sys.exit(1)
        
    model.eval()  # 평가 모드로 설정
    return model

def preprocess_image(image_path, width=IMG_WIDTH, height=IMG_HEIGHT):
    """
    이미지 전처리 함수
    Args:
        image_path: 이미지 파일 경로
        width: 입력 이미지 너비
        height: 입력 이미지 높이
    
    Returns:
        image_tensor: 전처리된 이미지 텐서
        original_image: 원본 이미지
        original_size: 원본 이미지 크기
    """
    try:
        # 원본 이미지 로드
        original_image = Image.open(image_path).convert("RGB")
        # 원본 이미지 크기 저장
        original_size = original_image.size
        
        # 리사이즈 이미지 (모델 입력용)
        resized_image = original_image.resize((width, height))
        # 이미지를 텐서로 변환
        image_tensor = transforms.ToTensor()(resized_image)
        # 배치 차원 추가
        image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
        
        return image_tensor, original_image, original_size
    except Exception as e:
        print(f"이미지 전처리 중 오류 발생: {e}")
        return None, None, None

def detect_objects(model, image, confidence_threshold, nms_threshold, device):
    """
    이미지에서 객체를 탐지하는 함수
    Args:
        model: 딥러닝 모델
        image: 전처리된 이미지 텐서 (1, C, H, W)
        confidence_threshold: 신뢰도 임계값
        nms_threshold: NMS 임계값
        device: 연산 장치
    
    Returns:
        outputs: 모델의 출력
    """
    try:
        # 이미지를 장치로 이동
        image = image.to(device)
        
        # 모델 예측 수행 - no_grad로 메모리 효율성 향상
        with torch.no_grad():
            outputs = model(image)
        
        return outputs
    except Exception as e:
        print(f"객체 탐지 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

def post_process(outputs, img_size, conf_threshold=0.5, nms_threshold=0.4):
    """
    모델 출력을 후처리하여 바운딩 박스, 신뢰도, 클래스 ID를 반환하는 함수
    Args:
        outputs: 모델의 출력값들 리스트
        img_size: 원본 이미지 크기 (width, height)
        conf_threshold: 신뢰도 임계값
        nms_threshold: NMS 임계값
    
    Returns:
        boxes: 바운딩 박스 좌표 리스트 [(x, y, w, h), ...]
        confidences: 신뢰도 리스트
        class_ids: 클래스 ID 리스트
    """
    boxes = []
    confidences = []
    class_ids = []
    
    # 특징 맵 크기
    img_w, img_h = img_size
    
    # 두 개의 출력(큰 물체용, 작은 물체용)에 대해 처리
    for output_idx, output in enumerate(outputs):
        # 출력 크기 정보
        batch_size, channels, feature_height, feature_width = output.size()
        
        # 앵커 수와 클래스 수 계산
        num_anchors = 3  # 앵커 수
        num_classes = (channels // num_anchors) - 5  # 클래스 수
        
        # 각 특징 맵 셀의 크기 비율 
        stride_w = img_w / feature_width
        stride_h = img_h / feature_height
        
        # 출력 재구성
        output = output.view(batch_size, num_anchors, 5 + num_classes, feature_height, feature_width)
        output = output.permute(0, 1, 3, 4, 2).contiguous().view(batch_size, -1, 5 + num_classes)
        
        # 첫 번째 배치만 처리 (배치 크기 = 1)
        output = output[0]
        
        # 앵커 정의 - 모델과 동일하게 설정
        anchors = [[0.28, 0.22], [0.38, 0.48], [0.9, 0.78]] if output_idx == 0 else [[0.07, 0.15], [0.15, 0.11], [0.14, 0.29]]
        
        # 각 앵커에 대한 예측 처리
        for anchor_idx in range(num_anchors):
            # 특징 맵 크기에 대한 그리드 셀 좌표 생성
            grid_x = torch.arange(feature_width).repeat(feature_height, 1).view([1, 1, feature_height, feature_width]).float().to(output.device)
            grid_y = torch.arange(feature_height).repeat(feature_width, 1).t().view([1, 1, feature_height, feature_width]).float().to(output.device)
            
            # 앵커 박스 넓이와 높이 
            anchor_w = torch.FloatTensor([[anchors[anchor_idx][0]]]).to(output.device)
            anchor_h = torch.FloatTensor([[anchors[anchor_idx][1]]]).to(output.device)
            
            # 모든 그리드 셀에 대한 박스 좌표와 신뢰도, 클래스 확률 추출
            stride = anchor_idx * feature_height * feature_width
            end = stride + feature_height * feature_width
            
            box_xy = torch.sigmoid(output[stride:end, :2])  # 중심 좌표 (x, y) - 시그모이드 적용
            box_wh = torch.exp(output[stride:end, 2:4]) * torch.cat([anchor_w, anchor_h], 1)  # 너비와 높이 (w, h)
            box_conf = torch.sigmoid(output[stride:end, 4])  # 신뢰도 - 시그모이드 적용
            box_cls = torch.sigmoid(output[stride:end, 5:])  # 클래스 확률 - 시그모이드 적용
            
            # 그리드 셀 인덱스 변환
            cell_indices = torch.arange(feature_height * feature_width).to(output.device)
            x_cell = cell_indices % feature_width
            y_cell = cell_indices // feature_width
            
            # 상대 좌표를 이미지 좌표로 변환
            x = (box_xy[:, 0] + x_cell) * stride_w
            y = (box_xy[:, 1] + y_cell) * stride_h
            w = box_wh[:, 0] * stride_w
            h = box_wh[:, 1] * stride_h
            
            # 임계값 이상의 물체만 선택
            conf_mask = box_conf >= conf_threshold
            
            for idx in range(len(box_conf)):
                if not conf_mask[idx]:
                    continue
                
                confidence = box_conf[idx].item()
                
                # 클래스별 확률
                cls_conf, cls_id = box_cls[idx].max(0)
                class_id = cls_id.item()
                
                # 최종 신뢰도 = 객체 존재 신뢰도 * 클래스 신뢰도
                final_confidence = confidence * cls_conf.item()
                
                if final_confidence < conf_threshold:
                    continue
                
                # 이미지 중심에서 좌상단 좌표로 변환 (YOLO 포맷 -> 일반 포맷)
                center_x, center_y = x[idx].item(), y[idx].item()
                width, height = w[idx].item(), h[idx].item()
                
                # 중심 좌표에서 좌상단 좌표로 변환
                x_min = max(0, center_x - width / 2)
                y_min = max(0, center_y - height / 2)
                
                # 박스, 신뢰도, 클래스 ID 저장
                boxes.append((x_min, y_min, width, height))
                confidences.append(final_confidence)
                class_ids.append(class_id)
    
    # Non-Maximum Suppression 적용
    if len(boxes) > 0:
        # 중복 박스 제거를 위한 NMS 적용
        indices = non_max_suppression(boxes, confidences, nms_threshold)
        
        # 결과 수집
        result_boxes = [boxes[i] for i in indices]
        result_confidences = [confidences[i] for i in indices]
        result_class_ids = [class_ids[i] for i in indices]
        
        return result_boxes, result_confidences, result_class_ids
    else:
        return [], [], []

def calculate_iou(box1, box2):
    """IoU 계산"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = intersection / float(box1_area + box2_area - intersection + 1e-6)
    return iou

def draw_boxes(image, boxes, scores, classes, class_names, max_boxes=30, min_score=0.01, colors=None):
    """바운딩 박스 그리기 함수"""
    if colors is None:
        # 전역 클래스 색상 사용
        colors = CLASS_COLORS
    
    # print(f"사용 중인 색상 매핑: {colors}")
    # print(f"첫 번째 클래스 ID: {classes[0] if classes else 'None'}")
    
    # 이미지 복사본 만들기 - RGBA 모드로 변환하여 투명도 지원
    if image.mode != 'RGBA':
        draw_image = image.convert('RGBA')
    else:
        draw_image = image.copy()
    
    # 레이블 표시용 투명 오버레이 생성
    overlay = Image.new('RGBA', draw_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(draw_image)
    overlay_draw = ImageDraw.Draw(overlay)
    width, height = image.size
    
    # 바운딩 박스 상자 그리기
    for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
        if i >= max_boxes or score < min_score:
            break
            
        # 클래스 ID와 이름 가져오기
        class_id = int(cls)
        class_name = class_names[class_id]
        label = f"{class_name}: {score:.2f}"
        
        # 클래스 ID별 색상 설정
        box_color = colors.get(class_id, (255, 0, 0))  # 기본 빨간색
        
        try:
            # 박스 좌표 변환 (x, y, w, h) -> (x1, y1, x2, y2)
            x, y, w, h = box
            
            # 너무 작은 너비나 높이 값 보정 (최소 2 픽셀)
            w = max(2.0, w)
            h = max(2.0, h)
            
            # 중심점 기준으로 box 좌표 계산
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)
            
            # 좌표가 이미지 범위를 벗어나지 않도록 조정
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width - 1, x2)
            y2 = min(height - 1, y2)
            
            # x2와 y2가 x1, y1과 같거나 작으면 최소 1픽셀 차이 보장
            if x2 <= x1:
                x2 = min(width - 1, x1 + 2)
            if y2 <= y1:
                y2 = min(height - 1, y1 + 2)
            
            # 디버깅 출력 (첫 5개 박스)
            if i < 5:
                print(f"박스 {i}: ({x1}, {y1}, {x2}, {y2}), 클래스: {class_name}(ID:{class_id}), 점수: {score:.2f}, 색상: {box_color}")
            
            # 테두리 굵기 설정
            line_width = 2
            
            # 실제 바운딩 박스 그리기 - 두껍게 그려서 확실히 보이게 함
            # RGBA 색상으로 변환 (알파값 255 = 완전 불투명)
            rgba_box_color = box_color + (255,) if len(box_color) == 3 else box_color
            draw.rectangle([x1, y1, x2, y2], outline=rgba_box_color, width=line_width)
            
            # 텍스트 설정
            try:
                font = ImageFont.truetype("arial.ttf", 14)
            except IOError:
                font = ImageFont.load_default()
                
            # 텍스트 크기 측정
            text_bbox = font.getbbox(label)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # 라벨 패딩 추가
            padding = 2
            text_width += padding * 2
            text_height += padding * 2
            
            # 라벨 위치 설정
            if y1 - text_height >= 0:
                text_y = y1 - text_height
            else:
                text_y = y1
            
            # 라벨 배경 그리기 - 반투명 배경 (알파값 128 = 50% 불투명)
            label_bg_color = box_color + (100,) if len(box_color) == 3 else (box_color[0], box_color[1], box_color[2], 100)
            overlay_draw.rectangle(
                [x1, text_y, x1 + text_width, text_y + text_height],
                fill=label_bg_color
            )
            
            # 텍스트 그리기 (오버레이에)
            text_color = (255, 255, 255, 255)  # 흰색, 완전 불투명
            overlay_draw.text((x1 + padding, text_y + padding), label, fill=text_color, font=font)
        
        except Exception as e:
            print(f"박스 그리기 오류 (박스 {i}): {e}")
            print(f"문제가 된 박스 좌표: {box}")
            continue
    
    # 원본 이미지와 오버레이 합성하여 반투명 효과 생성
    result = Image.alpha_composite(draw_image, overlay)
    # 결과 이미지가 원래 모드로 변환 (일반적으로 'RGB')
    if image.mode != 'RGBA':
        result = result.convert(image.mode)
    
    return result

def check_overlap(rect1, rect2):
    """두 직사각형이 겹치는지 확인"""
    # rect = (x1, y1, x2, y2)
    return not (rect1[2] < rect2[0] or rect1[0] > rect2[2] or rect1[3] < rect2[1] or rect1[1] > rect2[3])

def save_boxes_to_txt(save_path, boxes, scores, classes, class_names, image_size):
    """
    바운딩 박스 결과를 텍스트 파일로 저장하는 함수
    포맷: <class_name> <confidence> <x_center> <y_center> <width> <height>
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        for box, score, cls_id in zip(boxes, scores, classes):
            class_name = class_names[int(cls_id)]
            x, y, w, h = box
            
            # 이미지 크기로 정규화된 값 (0~1 사이)
            norm_x = x / image_size[0]
            norm_y = y / image_size[1]
            norm_w = w / image_size[0]
            norm_h = h / image_size[1]
            
            # 형식: <class_name> <confidence> <x_center> <y_center> <width> <height>
            f.write(f"{class_name} {score:.6f} {norm_x:.6f} {norm_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")
            
            # 디버깅용: 원본 좌표도 주석으로 저장
            f.write(f"# class={class_name}, x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}\n")

def non_max_suppression(boxes, scores, threshold):
    """
    Non-Maximum Suppression을 수행하는 함수
    Args:
        boxes: 바운딩 박스 좌표 리스트 [(x, y, w, h), ...]
        scores: 신뢰도 점수 리스트
        threshold: IoU 임계값
    
    Returns:
        indices: NMS 후 남은 박스의 인덱스 리스트
    """
    if len(boxes) == 0:
        return []
    
    # 박스 좌표 형식 변환: (x, y, w, h) -> (x1, y1, x2, y2)
    x1 = np.array([box[0] for box in boxes])
    y1 = np.array([box[1] for box in boxes])
    w = np.array([box[2] for box in boxes])
    h = np.array([box[3] for box in boxes])
    x2 = x1 + w
    y2 = y1 + h
    
    # 점수 기준 내림차순 정렬
    indices = np.argsort(scores)[::-1]
    
    keep = []
    while indices.size > 0:
        # 가장 높은 점수의 박스 선택
        i = indices[0]
        keep.append(i)
        
        # 겹치는 정도(IoU) 계산
        xx1 = np.maximum(x1[i], x1[indices[1:]])
        yy1 = np.maximum(y1[i], y1[indices[1:]])
        xx2 = np.minimum(x2[i], x2[indices[1:]])
        yy2 = np.minimum(y2[i], y2[indices[1:]])
        
        w_overlap = np.maximum(0, xx2 - xx1)
        h_overlap = np.maximum(0, yy2 - yy1)
        
        # 교집합 영역
        overlap_area = w_overlap * h_overlap
        
        # 각 박스의 면적
        area_i = w[i] * h[i]
        area_others = w[indices[1:]] * h[indices[1:]]
        
        # IoU 계산
        iou = overlap_area / (area_i + area_others - overlap_area)
        
        # IoU가 임계값보다 작은 박스만 유지
        mask = iou <= threshold
        indices = indices[1:][mask]
    
    return keep

def visualize_dataset_samples(dataset_dir, model=None, device=None, num_samples=3, confidence=0.5, nms_thres=0.4):
    """
    디버깅을 위한 데이터셋 시각화 함수
    원본 이미지와 '_marked' 이미지를 나란히 표시하고 선택적으로 모델 예측도 보여줌
    
    Args:
        dataset_dir: 데이터셋 디렉토리 경로
        model: 선택적 모델 (객체 탐지용)
        device: 모델 실행 장치
        num_samples: 표시할 샘플 수
        confidence: 객체 탐지 신뢰도 임계값
        nms_thres: NMS 임계값
    """
    # 결과 저장 디렉토리
    debug_dir = os.path.join(os.path.dirname(dataset_dir), 'debug')
    os.makedirs(debug_dir, exist_ok=True)
    
    # 이미지 파일 리스트 가져오기
    all_files = [f for f in os.listdir(dataset_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # 원본 이미지와 마킹된 이미지 분리
    original_files = [f for f in all_files if '_marked' not in f]
    marked_files = [f for f in all_files if '_marked' in f]
    
    # 파일 수 제한
    if num_samples > 0:
        original_files = original_files[:num_samples]
    
    # 각 원본 이미지에 대해 처리
    for i, orig_file in enumerate(original_files):
        base_name = os.path.splitext(orig_file)[0]
        
        # 대응되는 마킹된 이미지 찾기
        marked_file = next((f for f in marked_files if base_name in f), None)
        if not marked_file:
            print(f"'{orig_file}'에 대응하는 마킹된 이미지를 찾을 수 없습니다.")
            continue
        
        # 이미지 경로
        orig_path = os.path.join(dataset_dir, orig_file)
        marked_path = os.path.join(dataset_dir, marked_file)
        
        # 이미지 로드
        orig_img = Image.open(orig_path).convert("RGB")
        marked_img = Image.open(marked_path).convert("RGB")
        
        # matplotlib 그림 설정
        # fig, ax = plt.subplots(1, 2 if model is None else 3, figsize=(15, 5))
        fig, ax = plt.subplots(2 if model is None else 3, 1)#, figsize=(15, 5))
        
        # 원본 이미지 표시
        ax[0].imshow(orig_img)
        ax[0].set_title(f"원본: {orig_file}")
        ax[0].axis('off')
        
        # 마킹된 이미지 표시
        ax[1].imshow(marked_img)
        ax[1].set_title(f"Ground Truth: {marked_file}")
        ax[1].axis('off')
        
        # 모델이 제공된 경우 예측 수행
        if model is not None:
            # 이미지 전처리
            input_tensor, _, original_size = preprocess_image(orig_path)
            
            # 객체 탐지 수행
            outputs = detect_objects(model, input_tensor, confidence, nms_thres, device)
            
            # 후처리
            boxes, confidences, class_ids = post_process(
                outputs, original_size, 
                conf_threshold=confidence, 
                nms_threshold=nms_thres
            )
            
            # 결과 시각화
            result_image = draw_boxes(
                orig_img.copy(), 
                boxes, 
                confidences, 
                class_ids, 
                CLASS_NAMES, 
                colors=CLASS_COLORS
            )
            
            # 예측 이미지 표시
            ax[2].imshow(result_image)
            ax[2].set_title(f"모델 예측 (객체: {len(boxes)}개)")
            ax[2].axis('off')
        
        # 그림 저장
        plt.tight_layout()
        plt.savefig(os.path.join(debug_dir, f"debug_{i+1}_{base_name}.png"))
        plt.close()
        
        print(f"디버그 이미지 저장됨: debug_{i+1}_{base_name}.png")
    
    print(f"총 {len(original_files)}개의 디버그 이미지가 {debug_dir} 디렉토리에 저장되었습니다.")

def main(args):
    # 결과 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 모델 로드
    model = load_model(args.weights, DEVICE)
    
    # 이미지 파일 리스트
    if os.path.isdir(args.img_dir):
        image_paths = [os.path.join(args.img_dir, f) for f in os.listdir(args.img_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg')) and '_marked' not in f]
        print(f"처리할 이미지 수: {len(image_paths)}")
        
        # 제외된 _marked 파일 확인
        excluded_files = [f for f in os.listdir(args.img_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg')) and '_marked' in f]
        if excluded_files:
            print(f"'_marked' 포함으로 제외된 파일 ({len(excluded_files)}개): {', '.join(excluded_files)}")
    else:
        if '_marked' in args.img_dir:
            print(f"경고: 파일명에 '_marked'가 포함된 이미지를 처리합니다: {args.img_dir}")
        image_paths = [args.img_dir]
    
    # 각 이미지에 대해 객체 감지 수행
    for image_path in image_paths:
        try:
            print(f"처리 중인 이미지: {image_path}")
            
            # 이미지 전처리
            input_tensor, original_image, original_size = preprocess_image(
                image_path, 
                width=args.img_width, 
                height=args.img_height
            )
            
            if input_tensor is None:
                print(f"이미지를 처리할 수 없습니다: {image_path}")
                continue
            
            # 객체 감지 수행
            outputs = detect_objects(model, input_tensor, args.confidence, args.nms_thres, DEVICE)
            
            if outputs is None:
                print(f"이미지 처리 중 오류 발생: {image_path}")
                continue
            
            # 후처리
            boxes, confidences, class_ids = post_process(
                outputs, original_size, 
                conf_threshold=args.confidence, 
                nms_threshold=args.nms_thres
            )
            
            print(f"탐지된 객체 수: {len(boxes)}")
            # print(f"클래스 ID: {class_ids}")
            
            # 결과 시각화 - 클래스 ID별 색상 명시적으로 전달
            result_image = draw_boxes(
                original_image, 
                boxes, 
                confidences, 
                class_ids, 
                CLASS_NAMES, 
                colors=CLASS_COLORS
            )
            
            # 결과 이미지 저장
            output_image_path = os.path.join(args.output_dir, os.path.basename(image_path))
            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
            result_image.save(output_image_path)
            print(f"결과 이미지 저장됨: {output_image_path}")
            
            # 바운딩 박스 정보 텍스트 파일로 저장
            txt_path = os.path.splitext(output_image_path)[0] + '.txt'
            save_boxes_to_txt(txt_path, boxes, confidences, class_ids, CLASS_NAMES, original_size)
            print(f"바운딩 박스 정보 저장됨: {txt_path}")
            
        except Exception as e:
            print(f"이미지 처리 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RF 스펙트럼 신호 탐지")
    parser.add_argument("--weights", default="checkpoints/spectrum_model_final.pth", help="모델 가중치 파일 경로")
    parser.add_argument("--img-dir", default='./dataset', help="이미지 디렉토리 경로")
    parser.add_argument("--output-dir", default="results", help="결과 저장 디렉토리")
    parser.add_argument("--batch-size", type=int, default=1, help="배치 크기")
    parser.add_argument("--confidence", type=float, default=CONF_THRESHOLD, help="객체 탐지 신뢰도 임계값")
    parser.add_argument("--nms-thres", type=float, default=NMS_THRESHOLD, help="NMS 임계값")
    parser.add_argument("--img-width", type=int, default=IMG_WIDTH,
                        help="모델 입력 이미지 너비")
    parser.add_argument("--img-height", type=int, default=IMG_HEIGHT,
                        help="모델 입력 이미지 높이")
    parser.add_argument("--debug", action="store_true", help="디버깅 모드 활성화")
    parser.add_argument("--num-debug-samples", type=int, default=3, help="디버깅에 사용할 샘플 수")
    args = parser.parse_args()

    # 이미지 크기 설정값 업데이트
    IMG_WIDTH = args.img_width
    IMG_HEIGHT = args.img_height
    
    # 디버깅 모드인 경우
    # python.exe detect.py --debug
    if args.debug:
        # 모델 로드
        model = load_model(args.weights, DEVICE) if os.path.exists(args.weights) else None
        
        if model is None:
            print(f"주의: 모델 없이 디버그 모드 실행 (원본 및 GT 이미지만 표시)")
            
        # 시각화 함수 호출
        visualize_dataset_samples(
            args.img_dir, 
            model=model, 
            device=DEVICE, 
            num_samples=args.num_debug_samples,
            confidence=args.confidence,
            nms_thres=args.nms_thres
        )
    else:
        main(args)