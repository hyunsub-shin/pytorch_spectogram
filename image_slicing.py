import cv2
import os
import glob
from tqdm import tqdm  # 진행률 표시를 위해 필요 (pip install tqdm)

def _parse_yolo_label_to_bbox(label_tokens):
    """
    지원 포맷
    - YOLO bbox: cls xc yc w h
    - YOLO seg:  cls x1 y1 x2 y2 ... (정규화 좌표, 0~1)

    Returns:
        (cls_int, xc_n, yc_n, w_n, h_n) or None (파싱 실패)
    """
    if not label_tokens:
        return None

    # bbox: 5개 토큰
    if len(label_tokens) == 5:
        try:
            cls_f, xc_n, yc_n, w_n, h_n = map(float, label_tokens)
            return int(cls_f), xc_n, yc_n, w_n, h_n
        except Exception:
            return None

    # seg: cls + (x,y)쌍이 반복되어야 함
    if len(label_tokens) >= 7:
        try:
            cls_int = int(float(label_tokens[0]))
            coords = list(map(float, label_tokens[1:]))
            if len(coords) % 2 != 0:
                return None
            xs = coords[0::2]
            ys = coords[1::2]
            if not xs or not ys:
                return None
            x1 = min(xs); x2 = max(xs)
            y1 = min(ys); y2 = max(ys)
            w_n = max(0.0, x2 - x1)
            h_n = max(0.0, y2 - y1)
            xc_n = (x1 + x2) / 2.0
            yc_n = (y1 + y2) / 2.0
            return cls_int, xc_n, yc_n, w_n, h_n
        except Exception:
            return None

    return None

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
                    parsed = _parse_yolo_label_to_bbox(label)
                    if parsed is None:
                        continue
                    cls, xc_n, yc_n, w_n, h_n = parsed
                    
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
                        if x_start <= pt_x <= x_end and y_start <= pt_y <= y_end:
                            # 타일 상대 좌표로 변환 및 타일 사이즈로 정규화
                            nx = (pt_x - x_start) / tile_size
                            ny = (pt_y - y_start) / tile_size
                            new_poly.append(f"{nx:.6f} {ny:.6f}")

                    # 점이 최소 2개(Skeletal) 또는 3개(Polygon) 이상일 때만 저장
                    if len(new_poly) >= 2:
                        tile_labels.append(f"{class_id} {' '.join(new_poly)}")

                # 객체가 있는 타일만 저장
                if tile_labels:
                    tile_img = image[y_start:y_end, x_start:x_end]
                    save_name = f"{img_name}_{x_start}_{y_start}"
                    
                    cv2.imwrite(os.path.join(out_img_path, f"{save_name}.jpg"), tile_img)
                    with open(os.path.join(out_label_path, f"{save_name}.txt"), 'w') as f_out:
                        f_out.write("\n".join(tile_labels))
                        
# --- 사용 설정 ---
input_images = "./datasets/synthetic/images"    # 원본 이미지 폴더
input_labels = "./datasets/synthetic/labels"    # 원본 라벨 폴더
output_folder = "./datasets/sliced_data"    # 결과 저장 폴더

# batch_slice_yolo(input_images, input_labels, output_folder, tile_size=128, overlap=0.3)
batch_slice_yolo_polygon(input_images, input_labels, output_folder, tile_size=2560, overlap=0.3)