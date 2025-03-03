# 스펙트럼 이미지 객체 감지 프로젝트

이 프로젝트는 PyTorch를 사용하여 스펙트럼 이미지에서 다양한 패턴과 객체를 감지하는 머신러닝 모델을 구현합니다. 해당 모델은 YOLO v4 형식의 라벨링 데이터를 사용하여 학습합니다.

## 특징

- YOLO v4 형식의 라벨링 데이터 지원
- 이미지 내 여러 객체 감지 가능
- 맞춤형 객체 감지 모델 아키텍처
- 실시간 추론 및 결과 시각화
- 학습 진행 상황 추적 및 시각화

## 요구사항

- Python 3.9.12
- PyTorch 2.5.1-cu121
- cuda 12.1
- torchvision
- matplotlib
- numpy
- pillow
- tqdm

## 프로젝트 구조

```
pytorch_spectrogram/
  ├── models/               			# 모델 정의
  │     └── model_MobileNetV3.py 	# 객체 감지 모델
  ├── dataset/             			# 데이터셋 디렉토리
  │     ├── train/          		# 학습 데이터
  │     └── test/            		# 테스트 데이터
  ├── dataset.py           			# 데이터셋 클래스
  ├── train_MobileNetV3.py           # 학습 스크립트
  ├── detect_MobileNetV3.py          # 추론 및 시각화 스크립트
  ├── requirements.txt     			# 필요한 패키지 목록
  └── README.md            			# 프로젝트 설명
```

## 데이터 형식

이 프로젝트는 YOLO v4 형식의 라벨링 데이터를 사용합니다:

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
python train_MobileNetV3.py
```

학습 과정에서 검사점(checkpoint)과 손실 곡선이 각각 `checkpoints/`와 `results/` 디렉토리에 저장됩니다.

### 추론 및 시각화

```bash
python detect.py --input [이미지 경로 또는 디렉토리] --weights [모델 가중치 경로]
```

예:
```bash
python detect.py --input dataset/test/sample.png --weights checkpoints/spectrum_model_final.pth
```

추가 옵션:
```
--conf-thres: 객체 신뢰도 임계값 (기본값: 0.25)
--nms-thres: NMS 임계값 (기본값: 0.45)
--output-dir: 결과 저장 디렉토리 (기본값: results/)
```

## 모델 아키텍처
1. **백본 네트워크**: 기본적인 특징 추출을 위한 컨볼루션 레이어
2. **피처 피라미드**: 여러 크기의 객체를 감지하기 위한, 서로 다른 스케일의 피처맵
3. **앵커 박스**: 다양한 형태의 객체를 감지하기 위한 사전 정의된 박스
4. **헤드**: 클래스 분류, 객체 신뢰도, 바운딩 박스 조정을 위한 레이어


## 참고 자료
- [논문](https://www.mdpi.com/2306-5729/7/12/168)
- [dataset](https://fordatis.fraunhofer.de/handle/fordatis/287)
