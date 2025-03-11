# 스펙트럼 이미지 객체 감지 프로젝트

이 프로젝트는 PyTorch를 사용하여 스펙트럼 이미지에서 다양한 패턴과 객체를 감지하는 머신러닝 모델을 구현합니다. 해당 모델은 YOLO v4 형식의 라벨링 데이터를 사용하여 학습합니다.

## Prepare_data

- python prepare_data.py synthetic --drone (drone signal image) --back (background image) --output (output dir) --num (생성개수)

## 요구사항

certifi==2025.1.31
charset-normalizer==3.4.1
colorama==0.4.6
contourpy==1.3.0
cycler==0.12.1
filelock==3.13.1
fonttools==4.56.0
fsspec==2024.6.1
idna==3.10
importlib_resources==6.5.2
Jinja2==3.1.4
kiwisolver==1.4.7
MarkupSafe==2.1.5
matplotlib==3.9.4
mpmath==1.3.0
networkx==3.2.1
numpy==1.26.3
opencv-python==4.11.0.86
packaging==24.2
pandas==2.2.3
pillow==11.0.0
psutil==7.0.0
py-cpuinfo==9.0.0
pyparsing==3.2.1
python-dateutil==2.9.0.post0
pytz==2025.1
PyYAML==6.0.2
requests==2.32.3
scipy==1.13.1
seaborn==0.13.2
six==1.17.0
sympy==1.13.1
torch==2.5.1+cu121
torchaudio==2.5.1+cu121
torchvision==0.20.1+cu121
tqdm==4.67.1
typing_extensions==4.12.2
tzdata==2025.1
ultralytics==8.3.86
ultralytics-thop==2.0.14
urllib3==2.3.0
zipp==3.21.0

## 프로젝트 구조

```
pytorch_spectrogram/
  ├── datasets/             		# 데이터셋 디렉토리
  │     ├── images/          		# 이미지 디렉토리
  │   	 │	    ├── train/          		# 학습 데이터
  │     │		└── val/         		# 검증 데이터
  │     └── labels/            		# 라벨 디렉토리
  │   	 	    ├── train/          		# 학습 라벨
  │     		└── val/         		# 검증 라벨  
  ├── drone_dataset.yaml			# yaml 파일
  ├── prepare_data.py           	# 데이터셋 준비
  ├── yolov5_predict.py          # 예측 스크립트
  ├── yolov5_train.py          	# 학습 스크립트
  ├── requirements.txt     		# 필요한 패키지 목록
  └── README.md            		# 프로젝트 설명
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

