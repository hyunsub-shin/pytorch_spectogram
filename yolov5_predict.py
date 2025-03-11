import torch
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import numpy as np

CLASS_NAMES: ['WiFi', 'collision', 'BT']

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
            cv2.putText(img, f'{CLASS_NAMES[class_id]}: {confidence:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(255, 0, 0), thickness=2)

    # 결과 이미지 저장
    cv2.imwrite(result_image_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # OpenCV는 BGR로 저장

    return filtered_results.cpu().numpy()  # CUDA 텐서를 CPU로 이동 후 NumPy 배열로 변환


if __name__ == "__main__":
    image_path = 'spectrogram_training_data_20221006/result_frame_138847877310877880_bw_25E+6.png'  # 예측할 이미지 경로
    # image_path = 'drone_data/result_frame_138848079307226420_bw_125E+6.png'  # 예측할 이미지 경로
    save_directory = 'spectrogram_training_data_20221006'  # 결과를 저장할 사용자 지정 폴더
    # weights='last.pt'
    weights='best.pt'
    confidence_threshold = 0.5  # 신뢰도 임계값 설정
    predictions = predict(image_path, save_dir=save_directory, weights=weights, confidence_threshold=confidence_threshold)
    print(f'\npredictions.data \n{predictions.data}')  # 예측 결과 출력
   
    
# import cv2
# from ultralytics import YOLO

# model = YOLO('best.pt').to('cuda')
# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("웹캠을 열 수 없습니다.")
#     exit()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("프레임을 읽을 수 없습니다.")
#         break

#     results = model(frame)
#     annotated_frame = results[0].plot()

#     cv2.imshow('YOLOv8 Real-Time Detection', annotated_frame)

#     # 1ms 대기 후 ESC 키로 종료 (키보드 입력 확인)
#     if cv2.waitKey(1) & 0xFF == 27: 
#         break

# cap.release()
# cv2.destroyAllWindows()
