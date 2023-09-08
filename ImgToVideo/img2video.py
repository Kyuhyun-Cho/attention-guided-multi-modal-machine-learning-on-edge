import cv2
from datetime import datetime
import os, sys

def create_video_from_pngs(input_folder, output_video_path, frame_rate=30):
    # 폴더 내의 모든 파일 리스트를 얻습니다.
    file_list = sorted(os.listdir(input_folder))

    # 파일 리스트 중에서 png 파일들만 추려냅니다.
    png_files = [f for f in file_list if f.lower().endswith('.png')]

    # 비디오의 프레임 크기를 첫 번째 이미지를 기준으로 설정합니다.
    first_image_path = os.path.join(input_folder, png_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape

    # 비디오 코덱을 설정합니다.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 코덱을 사용합니다.

    # 비디오를 저장하기 위한 VideoWriter 객체를 생성합니다.
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    # 각 이미지를 비디오에 추가합니다.
    for png_file in png_files:
        image_path = os.path.join(input_folder, png_file)
        image = cv2.imread(image_path)

        # 이미지를 비디오에 추가합니다.
        out.write(image)

    # 사용한 자원을 해제합니다.
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_folder = sys.argv[1]  # png 파일들이 저장된 폴더 경로로 바꾸세요.
    output_video_path = './output/' +  datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.mp4' # 저장할 비디오 파일 경로로 바꾸세요.

    create_video_from_pngs(input_folder, output_video_path)

    print(output_video_path)
