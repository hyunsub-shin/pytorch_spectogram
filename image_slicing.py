import cv2
import os
import glob
from tqdm import tqdm  # 진행률 표시를 위해 필요 (pip install tqdm)

def batch_slice_yolo(input_img_dir, input_label_dir, output_dir, tile_size=1024, overlap=0.1):
    # 1. 출력 경로 설정 및 생성
    out_img_path = os.path.join(output_dir, 'images')
    out_label_path = os.path.join(output_dir, 'labels')
    os.makedirs(out_img_path, exist_ok=True)
    os.makedirs(out_label_path, exist_ok=True)

    # 2. 이미지 파일 목록 가져오기 (jpg, png 등 지원)
    img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    img_files = []
    for ext in img_extensions:
        img_files.extend(glob.glob(os.path.join(input_img_dir, ext)))

    print(f"총 {len(img_files)}개의 파일을 찾았습니다. 슬라이싱을 시작합니다.")

    # 3. 개별 파일 처리 루프
    for img_path in tqdm(img_files):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(input_label_dir, f"{img_name}.txt")

        # 라벨 파일이 없는 경우 건너뜀
        if not os.path.exists(label_path):
            continue

        # 이미지 로드
        image = cv2.imread(img_path)
        if image is None: continue
        h, w, _ = image.shape
        
        # 라벨 읽기
        with open(label_path, 'r') as f:
            labels = [line.strip().split() for line in f.readlines()]

        step = int(tile_size * (1 - overlap))

        # 4. 슬라이싱 (가로/세로)
        for y in range(0, h, step):
            for x in range(0, w, step):
                # 경계 보정 (이미지 끝을 넘지 않도록)
                x_end = min(x + tile_size, w)
                y_end = min(y + tile_size, h)
                x_start = max(0, x_end - tile_size)
                y_start = max(0, y_end - tile_size)

                tile_labels = []
                for label in labels:
                    cls, xc_n, yc_n, w_n, h_n = map(float, label)
                    
                    # 픽셀 좌표 복원
                    abs_xc, abs_yc = xc_n * w, yc_n * h
                    abs_w, abs_h = w_n * w, h_n * h
                    
                    x1, y1 = abs_xc - abs_w/2, abs_yc - abs_h/2
                    x2, y2 = abs_xc + abs_w/2, abs_yc + abs_h/2

                    #####################################################################
                    # 타일 영역 안에 있는지 확인(일부만 있어도 포함)
                    #####################################################################
                    if x1 < x_end and x2 > x_start and y1 < y_end and y2 > y_start:
                        # 타일 내 좌표로 변환 및 클리핑
                        nx1 = max(x1, x_start) - x_start
                        ny1 = max(y1, y_start) - y_start
                        nx2 = min(x2, x_end) - x_start
                        ny2 = min(y2, y_end) - y_start
                        
                        # 새로운 YOLO 정규화 좌표
                        nxc = (nx1 + nx2) / 2 / tile_size
                        nyc = (ny1 + ny2) / 2 / tile_size
                        nw = (nx2 - nx1) / tile_size
                        nh = (ny2 - ny1) / tile_size
                        
                        tile_labels.append(f"{int(cls)} {nxc:.6f} {nyc:.6f} {nw:.6f} {nh:.6f}")
                    #####################################################################

                    #####################################################################
                    # # 변경: 객체 전체가 타일 안에 완전히 들어와야만 포함
                    #####################################################################
                    # if x1 >= x_start and x2 <= x_end and y1 >= y_start and y2 <= y_end:
                    #     # 타일 내 좌표로 변환
                    #     nx1 = x1 - x_start
                    #     ny1 = y1 - y_start
                    #     nx2 = x2 - x_start
                    #     ny2 = y2 - y_start
                        
                    #     # 새로운 YOLO 정규화 좌표 생성
                    #     nxc = (nx1 + nx2) / 2 / tile_size
                    #     nyc = (ny1 + ny2) / 2 / tile_size
                    #     nw = (nx2 - nx1) / tile_size
                    #     nh = (ny2 - ny1) / tile_size
                        
                    #     tile_labels.append(f"{int(cls)} {nxc:.6f} {nyc:.6f} {nw:.6f} {nh:.6f}")
                    #####################################################################

                # 객체가 있는 타일만 저장 (배경만 있는 타일 제외로 데이터 효율화)
                if tile_labels:
                    tile_img = image[y_start:y_end, x_start:x_end]
                    suffix = f"_{x_start}_{y_start}"
                    
                    save_name = f"{img_name}{suffix}"
                    cv2.imwrite(os.path.join(out_img_path, f"{save_name}.jpg"), tile_img)
                    with open(os.path.join(out_label_path, f"{save_name}.txt"), 'w') as f:
                        f.write("\n".join(tile_labels))

# --- 사용 설정 ---
input_images = "./datasets/synthetic"    # 원본 이미지 폴더
input_labels = "./datasets/synthetic"    # 원본 라벨 폴더
output_folder = "./datasets/sliced_data"    # 결과 저장 폴더

batch_slice_yolo(input_images, input_labels, output_folder, tile_size=2560, overlap=0.3)