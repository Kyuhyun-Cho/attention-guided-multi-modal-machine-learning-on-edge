import sys
from ultralytics import YOLO
import cv2
from PIL import Image

model_1 = sys.argv[1]
# model_2 = sys.argv[2]
img_path = sys.argv[2]

model = YOLO("../runs/detect/train_" + model_1 + "_with_original/weights/best.pt")
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam

# from ndarray
img = cv2.imread(img_path)
results = model.predict(source=img, save=True, save_txt=True)
# results = model(img_path)

cv2.imshow("YOLOv8", results[0].plot())
cv2.waitKey(0)
cv2.destroyAllWindows()
# print(results[0].boxes)

print("Number of bbox:", results[0].boxes.shape[0])



# model = YOLO("../runs/detect/train_" + model_2 + "_with_original/weights/best.pt")
# # accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam

# # from ndarray
# im2 = cv2.imread(img_path)
# results = model.predict(source=im2, save=True, save_txt=True)

# # print(results[0].boxes)
# print("Number of bbox:", results[0].boxes.shape[0])
