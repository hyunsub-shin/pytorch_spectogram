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
        patience=patience,
        # lr0 = lr0,      # 추가 학습시 적용
        # freeze=10,       # 0~10번 레이어(Backbone 일부)를 고정
        # overlap_mask=True, # 세그멘테이션 성능 향상

        augment=True,    # 데이터 증강 활성화
        # ------ 세부 증강 설정 ------
        mosaic=0.0,      # 작은 객체 학습 강화 미적용(default: 1.0) <<== 해볼것(현재 1.0으로 학습)
        close_mosaic=10, # 마지막 10 에포크에서는 Mosaic을 끄고 학습
        scale=0.0, # 이미지 크기 변경 안함(default: 0.5)
        flipud=0.0, # 상하 뒤집기 미적용(default: 0.0)
        fliplr=0.0, # 좌우 뒤집기 미적용(default: 0.5)
        erasing=0.0, # 지우개 증강 미적용(default: 0.4)
        # --------------------------
        verbose=True     # 진행률 표시
    )

    print("Training completed.!!!!")


if __name__ == "__main__":
    img_size = 2560 #(1280) #첫데이터셋(206, 889) #(h, w)
    batch = 2
    epochs = 80
    patience = 15

    # # Segmentation
    # weights = 'yolo11n-seg.pt' # polygon label(segmentation)
    # weights = 'best-seg_resize_base_add_slice.pt'

    # # Detection
    # weights = 'yolo11n.pt' # box label(detection)
    # weights = 'best-det_resize_base.pt'

    # # yolov8/v11 P2 Layer
    weights = 'yolov8s-p2.yaml' # P2 레이어를 추가 1/4 해상도 단계에서 탐지를 수행
    # weights = 'yolov11n-p2_custom.yaml' # v11n P2 레이어를 추가 1/4 해상도 단계에서 탐지를 수행

    data_yaml = 'drone_dataset.yaml'
    # lr0 = 0.001 # 추가 학습시 적용

    main()
    
