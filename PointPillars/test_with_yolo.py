import matplotlib
matplotlib.use('Agg')

import argparse
import cv2
import numpy as np
import os
import torch
import pdb

from utils import setup_seed, read_points, read_calib, read_label, \
    keep_bbox_from_image_range, keep_bbox_from_lidar_range, vis_pc, \
    vis_img_3d, bbox3d2corners_camera, points_camera2image, \
    bbox_camera2lidar
from model import PointPillars

from ultralytics import YOLO

def point_range_filter(pts, point_range=[0, -39.68, -3, 69.12, 39.68, 1]):
    '''
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    point_range: [x1, y1, z1, x2, y2, z2]
    '''
    flag_x_low = pts[:, 0] > point_range[0]
    flag_y_low = pts[:, 1] > point_range[1]
    flag_z_low = pts[:, 2] > point_range[2]
    flag_x_high = pts[:, 0] < point_range[3]
    flag_y_high = pts[:, 1] < point_range[4]
    flag_z_high = pts[:, 2] < point_range[5]
    keep_mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
    pts = pts[keep_mask]
    return pts 

def inject_pcd2img(pc, img, calib_path, bbox_x1,bbox_y1, bbox_x2, bbox_y2):
    pc = pc[pc[:, 3] != 0]

    with open(calib_path, 'r') as f:
        lines = f.readlines()

    P2 = np.fromstring(lines[2].split(':')[1], sep=' ')
    R0_rect = np.fromstring(lines[4].split(':')[1], sep=' ')
    Tr_velo_to_cam = np.fromstring(lines[5].split(':')[1], sep=' ')

    P = P2.reshape(3, -1)
    R0 = np.eye(3)
    R0[:3, :3] = R0_rect.reshape(3, -1)
    Tr = Tr_velo_to_cam.reshape(3,-1)

    img_h, img_w, _ = img.shape
    print(pc)
    XYZ1 = np.concatenate((pc[:, :3].T, np.ones((1, pc.shape[0]))), axis=0)
    R_Tr_XYZ1 = np.dot(R0, np.dot(Tr, XYZ1))
    R_Tr_XYZ1 = np.concatenate((R_Tr_XYZ1, np.ones((1, R_Tr_XYZ1.shape[1]))), axis=0)

    xy1 = np.dot(P, R_Tr_XYZ1)

    s = xy1[2, :]
    x = xy1[0, :] / s
    y = xy1[1, :] / s

    for i in range(len(s)):
        ix = int(x[i])
        iy = int(y[i])
        if s[i] < 0 or ix <= bbox_x1+5 or ix >= bbox_x2-5 or iy <= bbox_y1+5 or iy >= bbox_y2-5:
            continue

        img[iy, ix, :] = [0, 255, 0]
    
    return 0


def calculate_iou(camera_bboxes, lidar_bboxes):
    final_bboxes = torch.empty(0, 4)

    for camera_bbox in camera_bboxes:
        c_x1, c_y1, c_x2, c_y2 = camera_bbox

        for lidar_bbox in lidar_bboxes:
            l_x1, l_y1, l_x2, l_y2 = lidar_bbox


            # Calculate intersection area
            x_left = max(c_x1, l_x1)
            y_top = max(c_y1, l_y1)
            x_right = min(c_x2, l_x2)
            y_bottom = min(c_y2, l_y2)
            
            if x_right < x_left or y_bottom < y_top:
                # No intersection
                continue
            else:
                intersection_area = (x_right - x_left) * (y_bottom - y_top)

                # Calculate union area
                camera_area_box = (c_x2 - c_x1) * (c_y2 - c_y1)
                lidar_area_box = (l_x2 - l_x1) * (l_y2 - l_y1)
                union_area = camera_area_box + lidar_area_box - intersection_area
                
                # Calculate IoU
                iou = intersection_area / union_area 
                # print ("IoU:", iou)
                if (iou >= 0.5):
                    intersection_box = torch.tensor([[x_left, y_top, x_right, y_bottom]])
                    final_bboxes = torch.cat((final_bboxes, intersection_box), dim=0)
                    break
                
    return final_bboxes

               
def main(args):
    CLASSES = {
        'Pedestrian': 0, 
        'Cyclist': 1, 
        'Car': 2
        }
    LABEL2CLASSES = {v:k for k, v in CLASSES.items()}
    # print(LABEL2CLASSES[0], LABEL2CLASSES[1], LABEL2CLASSES[2])
    pcd_limit_range = np.array([0, -40, -3, 70.4, 40, 0.0], dtype=np.float32)

    if not args.no_cuda:
        model = PointPillars(nclasses=len(CLASSES)).cuda()
        model.load_state_dict(torch.load(args.ckpt))
    else:
        model = PointPillars(nclasses=len(CLASSES))
        model.load_state_dict(
            torch.load(args.ckpt, map_location=torch.device('cpu')))
    
    if not os.path.exists(args.pc_path):
        raise FileNotFoundError 
    pc = read_points(args.pc_path)
    pc = point_range_filter(pc)
    pc_torch = torch.from_numpy(pc)
    if os.path.exists(args.calib_path):
        calib_info = read_calib(args.calib_path)
    else:
        calib_info = None
    
    if os.path.exists(args.gt_path):
        gt_label = read_label(args.gt_path)
    else:
        gt_label = None

    if os.path.exists(args.img_path):
        img = cv2.imread(args.img_path, 1)
        final_img = img.copy()
    else:
        img = None


    # YOLO Detect
    yolo_model = YOLO("/home/kyuhyunc/YOLOv8/runs/detect/train_medium_with_original/weights/best.pt")
    results = yolo_model.predict(source=img)
    yolo_img = results[0].plot()
    cam_lidar_img = results[0].plot()

    # for r in  results[0].boxes.xyxy:
    #     print(r)

    model.eval()
    with torch.no_grad():
        if not args.no_cuda:
            pc_torch = pc_torch.cuda()
        
        result_filter = model(batched_pts=[pc_torch], 
                              mode='test')[0]
        
    if calib_info is not None and img is not None:
        tr_velo_to_cam = calib_info['Tr_velo_to_cam'].astype(np.float32)
        r0_rect = calib_info['R0_rect'].astype(np.float32)
        P2 = calib_info['P2'].astype(np.float32)

        image_shape = img.shape[:2]
        result_filter = keep_bbox_from_image_range(result_filter, tr_velo_to_cam, r0_rect, P2, image_shape)

    result_filter = keep_bbox_from_lidar_range(result_filter, pcd_limit_range)

    lidar_bboxes = result_filter['lidar_bboxes']
    labels, scores = result_filter['labels'], result_filter['scores']

    # vis_pc(pc, bboxes=lidar_bboxes, labels=labels)

    if calib_info is not None and img is not None:
        bboxes2d, camera_bboxes = result_filter['bboxes2d'], result_filter['camera_bboxes'] 
        bboxes_corners = bbox3d2corners_camera(camera_bboxes)
        image_points = points_camera2image(bboxes_corners, P2)
        img = vis_img_3d(img, image_points, labels, rt=True)
        
        ##########################################
        # output_path = './output.png'
        # cv2.imwrite(output_path, img)
        # print(f"Detected result saved to {output_path}")
        ##########################################

    if calib_info is not None and gt_label is not None:
        tr_velo_to_cam = calib_info['Tr_velo_to_cam'].astype(np.float32)
        r0_rect = calib_info['R0_rect'].astype(np.float32)

        dimensions = gt_label['dimensions']
        location = gt_label['location']
        rotation_y = gt_label['rotation_y']
        gt_labels = np.array([CLASSES.get(item, -1) for item in gt_label['name']])
        sel = gt_labels != -1
        gt_labels = gt_labels[sel]
        bboxes_camera = np.concatenate([location, dimensions, rotation_y[:, None]], axis=-1)
        gt_lidar_bboxes = bbox_camera2lidar(bboxes_camera, tr_velo_to_cam, r0_rect)
        bboxes_camera = bboxes_camera[sel]
        gt_lidar_bboxes = gt_lidar_bboxes[sel]

        gt_labels = [-1] * len(gt_label['name']) # to distinguish between the ground truth and the predictions
        
        pred_gt_lidar_bboxes = np.concatenate([lidar_bboxes, gt_lidar_bboxes], axis=0)
        pred_gt_labels = np.concatenate([labels, gt_labels])
        
        # vis_pc(pc, pred_gt_lidar_bboxes, labels=pred_gt_labels)

        if img is not None:
            bboxes_corners = bbox3d2corners_camera(bboxes_camera)
            image_points = points_camera2image(bboxes_corners, P2)
            gt_labels = [-1] * len(gt_label['name'])
            # img = vis_img_3d(img, image_points, gt_labels, rt=True)

        
        # 2D & 3D Object Detection
        for i  in range(len(bboxes2d)):
            color = [0, 0 ,0]
            color[labels[i]] = 255
            x1, y1, x2, y2 = bboxes2d[i].astype(int)
            cv2.rectangle(cam_lidar_img, (x1, y1), (x2, y2), color, 2) 
            
            label = LABEL2CLASSES[labels[i]] + '_LiDAR'
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX  , 0.7, 1)
            text_w, text_h = text_size

            cv2.rectangle(cam_lidar_img, (x1-1, y1-text_h-2), (x1+text_w+1, y1+2), color, -1) # text backgroud
            cv2.putText(cam_lidar_img, label, (x1, y1-1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1) # label text

            # inject_pcd2img(pc, yolo_img, args.calib_path, x1, y1, x2, y2)
            # inject_pcd2img(pc, img, args.calib_path, x1, y1, x2, y2)
            inject_pcd2img(pc, cam_lidar_img, args.calib_path, x1, y1, x2, y2)

        # Fusion Detection
        final_bboxes = calculate_iou(results[0].boxes.xyxy, bboxes2d)
        for final_bbox in final_bboxes:
            color = [128, 0, 256]
            x1, y1, x2, y2 = map(int, final_bbox)
            cv2.rectangle(final_img, (x1, y1), (x2, y2), color, 2) 
            inject_pcd2img(pc, final_img, args.calib_path, x1, y1, x2, y2)

        cv2.imshow("2D Object Detection", yolo_img)
        cv2.imshow("3D Object Detection", img)
        cv2.imshow("2D & 3D Object Detection", cam_lidar_img)
        cv2.imshow("Fusion Detection", final_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # if calib_info is not None and img is not None:
    #     cv2.imshow(f'{os.path.basename(args.img_path)}-3d bbox', img) 
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--ckpt', default='pretrained/epoch_160.pth', help='your checkpoint for kitti')
    parser.add_argument('--pc_path', help='your point cloud path')
    parser.add_argument('--calib_path', default='', help='your calib file path')
    parser.add_argument('--gt_path', default='', help='your ground truth path')
    parser.add_argument('--img_path', default='', help='your image path')
    parser.add_argument('--no_cuda', action='store_true',
                        help='whether to use cuda')
    args = parser.parse_args()

    main(args)
