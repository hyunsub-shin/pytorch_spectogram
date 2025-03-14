# 스펙트럼 이미지 객체 감지 프로젝트

이 프로젝트는 PyTorch를 사용하여 스펙트럼 이미지에서 다양한 패턴과 객체를 감지하는 머신러닝 모델을 구현합니다. 해당 모델은 YOLO v4 형식의 라벨링 데이터를 사용하여 학습합니다.

## Prepare_data

```bash
python prepare_data.py synthetic --drone (drone signal image) --back (background image) --output (output dir) --num (생성개수)
```

## 요구사항

```bash
requirements.txt
```

## 프로젝트 구조

```
pytorch_spectrogram/
  ├── datasets/				# 데이터셋 디렉토리
  │     ├── images/				# 이미지 디렉토리
  │     │     ├── train/			# 학습 데이터
  │     │     └── val/				# 검증 데이터
  │     └── labels/				# 라벨 디렉토리
  │            ├── train/			# 학습 라벨
  │            └── val/				# 검증 라벨  
  ├── drone_dataset.yaml		# yaml 파일
  ├── prepare_data.py		# 데이터셋 준비
  ├── yolov5_predict.py		# 예측 스크립트
  ├── yolov5_train.py		# 학습 스크립트
  ├── requirements.txt		# 필요한 패키지 목록
  └── README.md				# 프로젝트 설명
```

## 데이터 형식

이 프로젝트는 YOLO v5 형식의 라벨링 데이터를 사용합니다:

- 이미지 파일: PNG/JPG 형식 (예: `image.png`)
- 라벨 파일: 각 이미지에 대한 텍스트 파일 (예: `image.txt`)

라벨 파일의 각 줄은 다음 형식을 따릅니다:
```
class_id center_x center_y width height
```

예:
```
2 0.059792 0.5 0.074196 0.08
0 0.114045 0.5 0.009102 0.8
1 0.30029 0.5 0.051632 0.8
```

여기서:
- `class_id`: 객체의 클래스 ID (0부터 시작)
- `center_x`, `center_y`: 객체의 중심 좌표 (0-1 범위로 정규화)
- `width`, `height`: 객체의 너비와 높이 (0-1 범위로 정규화)

## 사용 방법

### 모델 학습

```bash
python yolov5_train.py
```

학습 과정에서 검사점(checkpoint)과 손실 곡선이 각각 `runs/detect` 디렉토리에 저장됩니다.

### 추론 및 시각화

```bash
python yolov5_predict.py
```

추가 옵션:
```
--confidence_threshold: 객체 신뢰도 임계값 (기본값: 0.25)

```

