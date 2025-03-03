import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        return F.leaky_relu(self.bn(self.conv(x)), 0.1)

class SpectrumModel(nn.Module):
    def __init__(self, num_classes=3, anchors=None, pretrained=True):
        super(SpectrumModel, self).__init__()
        self.num_classes = num_classes
        # 기본 앵커 박스 (수정 가능)
        self.anchors = anchors or [
            [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],  # 큰 특징맵용 앵커
            [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)]  # 작은 특징맵용 앵커
        ]
        self.num_anchors = len(self.anchors[0])
        
        # MobileNetV3-Large를 백본으로 사용
        if pretrained:
            weights = MobileNet_V3_Large_Weights.DEFAULT
            self.backbone = mobilenet_v3_large(weights=weights)
        else:
            self.backbone = mobilenet_v3_large(weights=None)
        
        # 중간 및 최종 특징 맵의 채널 수를 동적으로 계산
        # 임시 입력으로 백본을 실행하여 실제 채널 수 얻기
        dummy_input = torch.zeros(1, 3, 192, 1024)
        with torch.no_grad():
            # 중간 특징 맵(10번째 레이어)의 채널 수 계산
            x = dummy_input
            for i in range(10):
                x = self.backbone.features[i](x)
            self.mid_channels = x.shape[1]  # 중간 특징 맵의 실제 채널 수
            
            # 마지막 특징 맵의 채널 수 계산
            for i in range(10, len(self.backbone.features)):
                x = self.backbone.features[i](x)
            self.backbone_out_channels = x.shape[1]  # 마지막 특징 맵의 실제 채널 수
            
        print(f"중간 특징 맵 채널 수: {self.mid_channels}")
        print(f"마지막 특징 맵 채널 수: {self.backbone_out_channels}")

        # 감지 레이어 1 (큰 객체용)
        self.detect1 = nn.Sequential(
            ConvBlock(self.backbone_out_channels, 512, kernel_size=1, padding=0),
            ConvBlock(512, 1024, kernel_size=3, padding=1),
            ConvBlock(1024, 512, kernel_size=1, padding=0)
        )
        
        # 감지 헤드 1
        self.head1 = nn.Conv2d(512, self.num_anchors * (5 + num_classes), kernel_size=1, padding=0)
        
        # 업샘플링 및 연결을 위한 레이어
        self.conv_for_concat = ConvBlock(512, 256, kernel_size=1, padding=0)
        
        # 감지 레이어 2 (작은 객체용)
        self.detect2 = nn.Sequential(
            ConvBlock(256 + self.mid_channels, 256, kernel_size=3, padding=1),
            ConvBlock(256, 512, kernel_size=3, padding=1)
        )
        
        # 감지 헤드 2
        self.head2 = nn.Conv2d(512, self.num_anchors * (5 + num_classes), kernel_size=1, padding=0)
        
    def forward(self, x):
        # 백본 네트워크 특징 추출 (중간 및 마지막 특징맵)
        features = []
        
        # 중간 특징맵 추출
        mid_features = x
        for i in range(10):  # 첫 10개 레이어
            mid_features = self.backbone.features[i](mid_features)
        features.append(mid_features)  # 중간 특징맵 저장
        
        # 마지막 특징맵 추출
        for i in range(10, len(self.backbone.features)):
            mid_features = self.backbone.features[i](mid_features)
        features.append(mid_features)  # 마지막 특징맵 저장
        
        # 감지 레이어 1 (큰 객체용)
        x_large = self.detect1(features[-1])
        large_output = self.head1(x_large)
        
        # 업샘플링 및 연결
        x_up = self.conv_for_concat(x_large)
        
        # 중간 특징맵의 크기에 맞추기 위해 직접 크기 지정
        x_up = F.interpolate(x_up, size=(features[0].size(2), features[0].size(3)), mode='nearest')
        
        x_concat = torch.cat([x_up, features[0]], dim=1)
        
        # 감지 레이어 2 (작은 객체용)
        x_small = self.detect2(x_concat)
        small_output = self.head2(x_small)
        
        return [large_output, small_output]

    def compute_loss(self, outputs, targets, device):
        """
        YOLO 손실 함수 계산
        outputs: 모델의 출력 [large_output, small_output]
        targets: 정답 라벨 (batch_size, num_objects, 5) - [class_id, x, y, w, h]
        """
        lambda_coord = 1.0  # 5.0에서 감소
        lambda_noobj = 0.1  # 0.5에서 감소
        
        # 출력 크기
        batch_size = outputs[0].size(0)
        total_loss = torch.tensor(0.0).to(device)
        
        for output_idx, output in enumerate(outputs):
            # 출력 크기 정보 확인
            batch_size, n_ch, grid_h, grid_w = output.size()
            grid_size = max(grid_h, grid_w)  # 격자 크기 (높이와 너비 중 큰 값 사용)
            stride = 416 // grid_size  # 입력 이미지 크기에 따라 조정
            anchors = self.anchors[output_idx]
            
            # 출력 재구성 - 동적으로 크기 계산
            # 각 앵커마다 5+num_classes 개의 값이 있음: x, y, w, h, confidence, class_scores
            prediction = output.view(batch_size, self.num_anchors, 5 + self.num_classes, 
                                     grid_h, grid_w).permute(0, 1, 3, 4, 2).contiguous()
            
            # 좌표와 물체 신뢰도 추출
            x = torch.sigmoid(prediction[..., 0])
            y = torch.sigmoid(prediction[..., 1])
            w = prediction[..., 2]
            h = prediction[..., 3]
            conf = torch.sigmoid(prediction[..., 4])
            cls_pred = torch.sigmoid(prediction[..., 5:])
            
            # 그리드 생성
            grid_x = torch.arange(grid_w).repeat(grid_h, 1).view([1, 1, grid_h, grid_w]).float().to(device)
            grid_y = torch.arange(grid_h).repeat(grid_w, 1).t().view([1, 1, grid_h, grid_w]).float().to(device)
            
            # 그리드 차원 확장 (배치 크기와 앵커 수에 맞게)
            grid_x = grid_x.expand(batch_size, self.num_anchors, grid_h, grid_w)
            grid_y = grid_y.expand(batch_size, self.num_anchors, grid_h, grid_w)
            
            # 앵커 박스 크기 설정
            anchor_w = torch.FloatTensor(anchors).index_select(1, torch.LongTensor([0])).to(device)
            anchor_h = torch.FloatTensor(anchors).index_select(1, torch.LongTensor([1])).to(device)
            anchor_w = anchor_w.view(1, self.num_anchors, 1, 1).expand(batch_size, self.num_anchors, grid_h, grid_w)
            anchor_h = anchor_h.view(1, self.num_anchors, 1, 1).expand(batch_size, self.num_anchors, grid_h, grid_w)
            
            # 겹치는 앵커와 타겟을 매칭시키는 로직
            obj_mask = torch.zeros_like(conf).to(device)
            noobj_mask = torch.ones_like(conf).to(device)
            
            # 각 배치에 대한 클래스 타겟 저장
            target_cls = []
            
            for b in range(batch_size):
                if targets[b] is None or len(targets[b]) == 0:
                    continue
                    
                for t in targets[b]:
                    if t.sum() == 0:  # 타겟이 없는 경우
                        continue
                    
                    # 타겟 좌표를 그리드 셀 좌표로 변환
                    gx = t[1] * grid_w
                    gy = t[2] * grid_h
                    gw = t[3] * grid_w
                    gh = t[4] * grid_h
                    
                    # 그리드 셀 인덱스 (범위 내로 제한)
                    gi = min(max(int(gx), 0), grid_w - 1)
                    gj = min(max(int(gy), 0), grid_h - 1)
                    
                    # 가장 적합한 앵커 찾기
                    target_box = torch.FloatTensor([0, 0, gw, gh]).unsqueeze(0).to(device)
                    anchor_shapes = torch.FloatTensor([(0, 0, anchor_w[b, a, gj, gi], anchor_h[b, a, gj, gi]) 
                                                      for a in range(self.num_anchors)]).to(device)
                    
                    # IoU 기반 앵커 선택
                    anch_ious = self.bbox_iou(target_box, anchor_shapes)
                    best_anchor = anch_ious.argmax().item()
                    
                    # 마스크와 타겟 설정
                    if gi < grid_w and gj < grid_h:
                        obj_mask[b, best_anchor, gj, gi] = 1
                        noobj_mask[b, best_anchor, gj, gi] = 0
                        
                        # 클래스 타겟 저장
                        target_cls.append((b, best_anchor, gj, gi, t[0].int().item()))
            
            # 손실 계산
            # obj_mask가 비어있는지 확인 (타겟이 없는 경우)
            if obj_mask.sum() > 0:
                coord_loss = lambda_coord * (torch.mean((x[obj_mask == 1] - grid_x[obj_mask == 1]) ** 2) +
                                          torch.mean((y[obj_mask == 1] - grid_y[obj_mask == 1]) ** 2) +
                                          torch.mean((torch.exp(w[obj_mask == 1]) - anchor_w[obj_mask == 1]) ** 2) +
                                          torch.mean((torch.exp(h[obj_mask == 1]) - anchor_h[obj_mask == 1]) ** 2))
                
                obj_loss = torch.mean((conf[obj_mask == 1] - 1) ** 2)
            else:
                coord_loss = torch.tensor(0.0).to(device)
                obj_loss = torch.tensor(0.0).to(device)
                
            noobj_loss = lambda_noobj * torch.mean((conf[noobj_mask == 1]) ** 2)
            
            # 클래스 손실 (평균 사용)
            cls_loss = torch.tensor(0.0).to(device)
            if len(target_cls) > 0:
                for b, a, gj, gi, target_c in target_cls:
                    cls_loss += F.mse_loss(
                        cls_pred[b, a, gj, gi], 
                        F.one_hot(torch.tensor(target_c), self.num_classes).float().to(device),
                        reduction='mean'
                    )
                cls_loss = cls_loss / len(target_cls)
            
            # 총 손실
            loss = coord_loss + obj_loss + noobj_loss + cls_loss
            total_loss += loss
        
        return total_loss / batch_size
    
    def bbox_iou(self, box1, box2):
        """
        박스 간 IoU 계산
        """
        # 박스 형식: [0, 0, w, h]
        b1_x1, b1_y1 = torch.zeros(1).to(box1.device), torch.zeros(1).to(box1.device)
        b1_x2, b1_y2 = box1[:, 2], box1[:, 3]
        
        b2_x1, b2_y1 = torch.zeros(1).to(box2.device), torch.zeros(1).to(box2.device)
        b2_x2, b2_y2 = box2[:, 2], box2[:, 3]
        
        # 교차 영역 계산
        inter_rect_x1 = torch.max(b1_x1, b2_x1)
        inter_rect_y1 = torch.max(b1_y1, b2_y1)
        inter_rect_x2 = torch.min(b1_x2, b2_x2)
        inter_rect_y2 = torch.min(b1_y2, b2_y2)
        
        inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                     torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
        
        # 합집합 영역 계산
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
        
        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
        
        return iou
