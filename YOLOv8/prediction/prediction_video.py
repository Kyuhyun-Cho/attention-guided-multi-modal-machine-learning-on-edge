import sys
import cv2
import csv
from ultralytics import YOLO

def count_items_in_output(cls):
    # 항목별 개수를 저장할 딕셔너리 초기화
    count_dict = {
        'Car' : 0,
        'Van' : 0,
        'Truck' : 0,
        'Pedestrian' : 0,
        'Person_sitting' : 0,
        'Cyclist' : 0,
        'Tram' : 0,
        'Misc' : 0,
        'DontCare' : 0,
        'Total': 0
    }

    # 출력문을 공백을 기준으로 분리하여 항목별 개수 세기
    for object_cls in cls:
        object_name = 'DontCare'

        if object_cls == 0: object_name = 'Car'
        elif object_cls == 1: object_name = 'Van'
        elif object_cls == 2: object_name = 'Truck'
        elif object_cls == 3: object_name = 'Pedestrian'
        elif object_cls == 4: object_name = 'Person_sitting'
        elif object_cls == 5: object_name = 'Cyclist'
        elif object_cls == 6: object_name = 'Tram'
        elif object_cls == 7: object_name = 'Misc'
        elif object_cls == 8: object_name = 'DontCare'

        count_dict[object_name] += 1

    count_dict['Total'] = len(cls)

    return count_dict


if __name__ == "__main__":
    video_path = sys.argv[1]
    file_name = video_path.split('/')[-1].split('.')[0]

    frame_num = 0

    # Load the YOLOv8 model
    model = YOLO('../runs/detect/train_medium_with_original/weights/best.pt')

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Create and open the count.txt file in write mode
    with open('./output/' + file_name + '_count.csv', 'w', newline='') as count_file:
        fieldnames = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare' , 'Total']
        writer = csv.DictWriter(count_file, fieldnames=fieldnames)
        writer.writeheader()

        while True:
            # Read a frame from the video
            success, frame = cap.read()

            if success:
                # Run YOLOv8 inference on the frame
                results = model(frame)

                # Write the count to the count.txt file
                count_dict = count_items_in_output(results[0].boxes.cls)
                writer.writerow(count_dict)
                
                # Visualize the results on the frame
                annotated_frame = results[0].plot()

                # Display the annotated frame
                cv2.imshow('YOLOv8 Inference', annotated_frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # Rewind the video to the beginning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                break

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
