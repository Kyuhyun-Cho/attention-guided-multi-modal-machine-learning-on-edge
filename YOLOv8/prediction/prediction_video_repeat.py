import sys
import cv2
from ultralytics import YOLO

video_path = sys.argv[1]

# Load the YOLOv8 model
model = YOLO('../runs/detect/train_medium_with_original/weights/best.pt')

# Open the video file
cap = cv2.VideoCapture(video_path)

while True:
    # Check if the video capture object is opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        break

    # Loop through the video frames
    while True:
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = model(frame)
            # print(len(results[0].orig_img[1]))
            # boxes = results[0].boxes
            # box = boxes[0]  # returns one box
            # print("Box xyxy:", box.xyxy)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Rewind the video to the beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
