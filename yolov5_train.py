import os
import torch
from ultralytics import YOLO

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    print("\n=== 시스템 정보 ===")
    print(f"CUDA 사용 가능 여부: {torch.cuda.is_available()}")
    print(f"현재 PyTorch의 CUDA 버전: {torch.version.cuda}")
    print(f"PyTorch 버전: {torch.__version__}") # 예: '2.0.0+cu121'는 CUDA 12.1 버전을 지원
    print(f"사용 장치: {device}")

    if device.type == 'cuda':        
        print(f'현재 사용 중인 GPU: {torch.cuda.get_device_name(0)}')
        
        # CUDA 설정
        torch.backends.cudnn.benchmark = True # 속도 향상을 위한 설정
        torch.backends.cudnn.deterministic = True # 재현 가능성 확보
        torch.backends.cuda.matmul.allow_tf32 = True  # TensorFloat-32 사용        
        torch.cuda.empty_cache()
        # 메모리 할당 모드 설정
        # torch.cuda.set_per_process_memory_fraction(0.8)  # GPU 메모리의 80% 사용

    # YOLOv5 모델 로드
    model = YOLO(weights)

    # 모델 학습
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,  # 이미지 크기
        batch=batch,     # 배치 크기
        name='final',    # 결과 폴더 이름
        device=device,
        verbose=True     # 진행률 표시
    )

    print("Training completed.!!!!")


if __name__ == "__main__":
    img_size = (206, 889) # (h, w)
    batch = 2
    epochs = 20
    weights = 'yolov5su.pt'
    data_yaml = 'yolov5/drone_dataset.yaml'

    main()
