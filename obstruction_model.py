import os
import numpy as np
import cv2
import sys

def apply_obstructions(image, blur_count, min_side_length, max_intensity):
    h, w, _ = image.shape

    obstructions_info = []  # List to store obstructions' info
    
    for _ in range(blur_count):
        # Randomly generate obstruction position
        x = np.random.randint(0, h - min_side_length)
        y = np.random.randint(0, w - min_side_length)
        
        # Randomly generate obstruction size
        side_length = np.random.randint(min_side_length, min(h - x, w - y))
        
        # Generate Gaussian profile for intensity attenuation
        intensity_attenuation = np.zeros((side_length, side_length))
        center = side_length // 2
        for i in range(side_length):
            for j in range(side_length):
                distance_to_center = np.sqrt((i - center)**2 + (j - center)**2)
                intensity_attenuation[i, j] = max_intensity * np.exp(-0.5 * (distance_to_center / center)**2) 
        
        # Set class per noise
        class_level = 0
        if 12 <= side_length < 72: class_level = 0
        elif 72 <= side_length < 120: class_level = 1
        elif 120 <= side_length: class_level = 2

        # Append obstruction info to the list
        obstructions_info.append((x, y, side_length, class_level))
        
        # Apply obstruction by adjusting brightness
        for i in range(side_length):
            for j in range(side_length):
                intensity = intensity_attenuation[i, j]
                scaled_intensity = max_intensity - intensity  # Invert intensity to create desired effect
                if (scaled_intensity > 1):
                    scaled_intensity = 1
                image[x+i, y+j, :] = np.clip(image[x+i, y+j, :] * scaled_intensity, 0, 255).astype(np.uint8)
    
    return image, obstructions_info

# Parameters
blur_count = int(sys.argv[1])           # Number of obstructions
min_side_length = int(sys.argv[2])       # Minimum obstruction side length
max_intensity = float(sys.argv[3])      # Maximum intensity attenuation
image_dir = './YOLOv8/data_with_noise/selected_images'  # Replace with your image directory path
output_image_dir = './YOLOv8/data_with_noise/selected_images_noise'  # Replace with the directory where you want to save modified images
output_label_dir = './YOLOv8/data_with_noise/selected_images_noise_labels'  # Replace with the directory where you want to save label files

# Ensure the output directories exist
if not os.path.exists(output_image_dir):
    os.makedirs(output_image_dir)

if not os.path.exists(output_label_dir):
    os.makedirs(output_label_dir)

# Apply obstructions with Gaussian intensity profile to all images in the directory
image_files = os.listdir(image_dir)
image_files.sort()
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    image = cv2.imread(image_path)

    image_with_obstructions, obstructions_info = apply_obstructions(image.copy(), blur_count, min_side_length, max_intensity*3)
    
    # Save modified image to output directory
    output_image_path = os.path.join(output_image_dir, image_file)
    cv2.imwrite(output_image_path, image_with_obstructions)
    
    # Create label file
    label_file_path = os.path.join(output_label_dir, os.path.splitext(image_file)[0] + '.txt')
    with open(label_file_path, 'w') as label_file:
        for info in obstructions_info:
            class_label = info[3]
            x_center = (info[1] + info[2] / 2) / image.shape[1]
            y_center = (info[0] + info[2] / 2) / image.shape[0]
            width = info[2] / image.shape[1]
            height = info[2] / image.shape[0]
            label_file.write(f'{class_label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n')
    
    print(image_file)
