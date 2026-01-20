import torch
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import numpy as np

def video():
    # import cv2
    # from ultralytics import YOLO

    model = YOLO('yolov5su.pt').to('cuda')
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        cv2.imshow('YOLOv8 Real-Time Detection', annotated_frame)

        # 1ms 대기 후 ESC 키로 종료 (키보드 입력 확인)
        if cv2.waitKey(1) & 0xFF == 27: 
            break

    cap.release()
    cv2.destroyAllWindows()
    
def predict(image_path, save_dir, weights, confidence_threshold=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = "cpu"
    print(f"사용 장치: {device}")

    # YOLOv5 모델 로드 (학습된 가중치 사용)
    model = YOLO(weights).to(device)

    # 이미지 로드
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV는 BGR로 이미지를 로드하므로 RGB로 변환

    # 예측 수행
    with torch.no_grad():  # 기울기 계산 비활성화
        results = model(img)

    # 결과가 리스트인지 확인
    if isinstance(results, list):
        # 각 결과에서 바운딩 박스 정보 추출
        boxes = []
        for result in results:
            if hasattr(result, 'boxes'):
                boxes.append(result.boxes.xyxy.cpu().numpy())  # 바운딩 박스 좌표를 NumPy 배열로 변환

        # boxes가 비어 있지 않은 경우
        if boxes:
            boxes = np.concatenate(boxes)  # 모든 박스를 하나의 NumPy 배열로 결합
            print("Extracted bounding boxes:", boxes)
        else:
            print("No bounding boxes found.")
            return np.array([])  # 빈 배열 반환
    else:
        print("Results is not a list.")
        return np.array([])  # 빈 배열 반환

    # 예측 결과를 confidence threshold에 따라 필터링
    # 결과 필터링: 신뢰도가 threshold 이상인 경우만 표시
    if len(results) > 0:
        filtered_results = results[0].boxes[results[0].boxes.conf >= confidence_threshold]  # 신뢰도 필터링
    else:
        print("No predictions were made.")
        return np.array([])  # 빈 배열 반환

    # 결과 시각화
    if len(filtered_results) > 0:
        results[0].show()  # 첫 번째 결과를 화면에 표시
    else:
        print("No detections above the confidence threshold.")

    # 결과 저장 폴더가 없으면 생성
    os.makedirs(save_dir, exist_ok=True)

    # 결과를 지정한 폴더에 PNG 형식으로 저장
    result_image_path = os.path.join(save_dir, 'result.png')  # 저장할 파일 경로

    # 필터링된 결과를 기반으로 이미지 저장
    if len(filtered_results) > 0:
        # 필터링된 결과를 이미지에 그리기
        for box in filtered_results:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # 바운딩 박스 좌표
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color=(255, 0, 0), thickness=2)  # 바운딩 박스 그리기
            # 텐서를 정수로 변환하여 문자열로 변환
            class_id = int(box.cls[0].item())  # 클래스 ID
            confidence = box.conf.item()  # 신뢰도
            # 클래스 ID 범위 체크
            if 0 <= class_id < len(CLASS_NAMES):
                label = f'{CLASS_NAMES[class_id]}: {confidence:.2f}'
            else:
                label = f'Class{class_id}: {confidence:.2f}'  # 범위를 벗어나면 기본 형식 사용
            cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(255, 0, 0), thickness=2)
            
    # 결과 이미지 저장
    cv2.imwrite(result_image_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # OpenCV는 BGR로 저장

    return filtered_results.cpu().numpy()  # CUDA 텐서를 CPU로 이동 후 NumPy 배열로 변환

def predict_with_slicing(image_path, save_dir, weights, tile_size=2560, overlap=0.3, confidence_threshold=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 장치: {device}")

    # 모델 로드
    model = YOLO(weights).to(device)

    # 원본 이미지 로드
    original_img = cv2.imread(image_path)
    h, w, _ = original_img.shape
    
    # 최종 결과를 담을 리스트 (원본 좌표 기준)
    all_detections = []

    # 슬라이싱 파라미터 계산
    step = int(tile_size * (1 - overlap))

    # 1. 슬라이싱 루프 실행
    for y in range(0, h, step):
        for x in range(0, w, step):
            # 타일 경계 계산
            x_end = min(x + tile_size, w)
            y_end = min(y + tile_size, h)
            x_start = max(0, x_end - tile_size)
            y_start = max(0, y_end - tile_size)

            # 타일 자르기
            tile = original_img[y_start:y_end, x_start:x_end]
            
            # 모델 추론 (imgsz는 학습 시 설정한 1280 추천)
            results = model.predict(
                tile, 
                imgsz=1280, 
                conf=confidence_threshold, 
                verbose=False, 
                show_boxes=True     # 박스 결과 출력 설정
            #    masks=True      # 마스크(폴리곤) 결과 출력 설정 (Segmentation 모델일 경우에만 작동)
            )

            # 2. 검출된 객체들을 원본 좌표로 변환 (Mask 추가 버전)
            for result in results:
                boxes = result.boxes
                masks = result.masks  # 폴리곤(마스크) 데이터 가져오기

                for i, box in enumerate(boxes):
                    # 타일 내 박스 좌표
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].item()
                    cls = box.cls[0].item()

                    # 원본 좌표로 복원 (박스)
                    abs_x1 = xyxy[0] + x_start
                    abs_y1 = xyxy[1] + y_start
                    abs_x2 = xyxy[2] + x_start
                    abs_y2 = xyxy[3] + y_start

                    # 폴리곤 좌표 복원
                    current_mask = None
                    if masks is not None:
                        # 해당 객체의 폴리곤 좌표 (N, 2)
                        segments = masks.xy[i] 
                        # 타일 좌표에 Offset을 더해 원본 좌표로 변환
                        abs_segments = segments.copy()
                        abs_segments[:, 0] += x_start
                        abs_segments[:, 1] += y_start
                        current_mask = abs_segments

                    # 박스, 점수, 클래스, 그리고 마스크를 함께 저장
                    all_detections.append({
                        'box': [abs_x1, abs_y1, abs_x2, abs_y2],
                        'conf': conf,
                        'cls': cls,
                        'mask': current_mask
                    })

            # 3. 중복 제거 (NMS)
            if len(all_detections) > 0:
                # .float()를 붙여서 두 텐서를 모두 float32 형식으로 맞춥니다.
                boxes_only = torch.tensor([d['box'] for d in all_detections]).float()
                scores_only = torch.tensor([d['conf'] for d in all_detections]).float()

                # GPU를 사용 중이라면 동일한 장치(device)로 이동시키는 작업도 필요할 수 있습니다.
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                boxes_only = boxes_only.to(device)
                scores_only = scores_only.to(device)
                
                import torchvision
                keep_indices = torchvision.ops.nms(boxes_only, scores_only, iou_threshold=0.4)
                
                # NMS에서 살아남은 최종 결과들만 필터링
                final_detections = [all_detections[i] for i in keep_indices.cpu().numpy()]
            else:
                final_detections = []

    # 4. 시각화 및 저장 (Polygon/Mask 지원 버전)
    display_img = original_img.copy()
    
    # 마스크를 그릴 투명 레이어 생성
    overlay = display_img.copy()
    
    for det in final_detections:
        box = det['box']
        conf = det['conf']
        cls = det['cls']
        mask = det['mask']  # 원본 좌표로 복원된 폴리곤 점들 (N, 2)
        
        # 색상 설정 (BGR: 빨강색)
        color = (0, 0, 255)
        
        # 4-1. 폴리곤 마스크 그리기 (색칠)
        if mask is not None:
            # 좌표를 정수형으로 변환
            pts = mask.astype(np.int32)
            # 폴리곤 내부 채우기
            cv2.fillPoly(overlay, [pts], color)
            # 외곽선 그리기
            cv2.polylines(display_img, [pts], True, color, 3)

        # 4-2. 박스 및 라벨 그리기
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
        
        cls_int = int(cls)
        label = f"{CLASS_NAMES[cls_int] if cls_int < len(CLASS_NAMES) else 'Obj'} {conf:.2f}"
        cv2.putText(display_img, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)

    # 4-3. 투명도 적용 (원본 60% + 마스크 40%)
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, display_img, 1 - alpha, 0, display_img)

    # 결과 저장
    os.makedirs(save_dir, exist_ok=True)
    img_filename = os.path.basename(image_path)
    save_filename = f"result_{img_filename}"
    result_path = os.path.join(save_dir, save_filename)
    cv2.imwrite(result_path, display_img)
    print(f"✅ 폴리곤 결과 저장 완료: {result_path}")

    return final_detections

if __name__ == "__main__":
    # drone_dataset.yaml의 names와 일치시켜야 함
    CLASS_NAMES = ['mini', 'autelevo']
    # CLASS_NAMES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 
    #     'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 
    #     57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
    
    image_path = 'datasets/drone_data/img/mini2_mini3_9.png'  # 예측할 이미지 경로
    # image_path = 'drone_data/result_frame_138848079307226420_bw_125E+6.png'  # 예측할 이미지 경로
    save_directory = 'datasets'  # 결과를 저장할 사용자 지정 폴더
    # weights='last.pt'
    weights='best.pt'#'yolov5su'#
    confidence_threshold = 0.5  # 신뢰도 임계값 설정

    ###############################################################################
    # # 원본 이미지 추론 실행
    ###############################################################################
    # predictions = predict(
    #     image_path, 
    #     save_dir=save_directory, 
    #     weights=weights, 
    #     confidence_threshold=confidence_threshold
    # )
    # print(f'\npredictions.data \n{predictions.data}')  # 예측 결과 출력
    ###############################################################################
   
    ###############################################################################
    # 슬라이싱 이미지 추론 실행
    ###############################################################################
    predictions = predict_with_slicing(
        image_path, 
        save_dir=save_directory, 
        weights=weights, 
        tile_size=2560,   # 학습 시 설정한 크기와 동일하게
        overlap=0.3,      # 학습 시 설정한 중첩률
        confidence_threshold=confidence_threshold
    )
    print(f"검출된 총 객체 수: {len(predictions)}")
    ###############################################################################

    # camera capture test
    # video()

