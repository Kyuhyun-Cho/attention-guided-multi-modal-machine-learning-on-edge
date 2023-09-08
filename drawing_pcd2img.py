import torch
import numpy as np
import matplotlib.pyplot as plt

# Read calibration file and parse the matrices
with open('./PointPillars/kitti/training/calib/000001.txt', 'r') as f:
    lines = f.readlines()
    
# P0 = np.fromstring(lines[0].split(':')[1], sep=' ')
# P1 = np.fromstring(lines[1].split(':')[1], sep=' ')
P2 = np.fromstring(lines[2].split(':')[1], sep=' ')
# P3 = np.fromstring(lines[3].split(':')[1], sep=' ')
R0_rect = np.fromstring(lines[4].split(':')[1], sep=' ')
Tr_velo_to_cam = np.fromstring(lines[5].split(':')[1], sep=' ')
# Tr_imu_to_velo = np.fromstring(lines[6].split(':')[1], sep=' ')

P = P2.reshape(3, -1)

R0 = np.eye(3)
R0[:3, :3] = R0_rect.reshape(3, -1)

Tr = Tr_velo_to_cam.reshape(3,-1)

# Read image
img = plt.imread('./PointPillars/kitti/training/image_2/000001.png')
plt.figure(figsize=(12, 10)) 
plt.imshow(img)
plt.show()

# Read LiDAR data
data = np.fromfile('./PointPillars/kitti/training/velodyne_reduced/000001.bin', dtype=np.float32).reshape(-1, 4)
data = data[data[:, 3] != 0]
# Mapping to image
img_mapped = img.copy()

img_h, img_w, _ = img.shape

XYZ1 = np.concatenate((data[:, :3].T, np.ones((1, data.shape[0]))), axis=0)
R_Tr_XYZ1 = np.dot(R0, np.dot(Tr, XYZ1))
R_Tr_XYZ1 = np.concatenate((R_Tr_XYZ1, np.ones((1, R_Tr_XYZ1.shape[1]))), axis=0)

xy1 = np.dot(P, R_Tr_XYZ1)

s = xy1[2, :]
x = xy1[0, :] / s
y = xy1[1, :] / s

for i in range(len(s)):
    ix = int(x[i])
    iy = int(y[i])
    if s[i] < 0 or ix < 0 or ix >= img_w or iy < 0 or iy >= img_h:
        continue

    img_mapped[iy, ix, :] = [0, 255, 0]

plt.figure(figsize=(12, 10)) 
plt.imshow(img_mapped)
plt.show()