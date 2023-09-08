import torch
from ultralytics import YOLO  # YOLOv8 모델 클래스의 임포트를 가정합니다.
import sys

pretrained_model = sys.argv[1]
# YOLOv8 모델을 초기화합니다.
model = YOLO('./runs/detect/train_'+ pretrained_model +'_with_original/weights/best.pt')

torch_model = model.model
# 모델의 파라미터 수를 계산합니다.
total_params = sum(p.numel() for p in torch_model.parameters())
print(pretrained_model, f"total parameters: {total_params}")
