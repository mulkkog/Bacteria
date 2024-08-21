import os
import numpy as np
import matplotlib.pyplot as plt
import wandb
import io
from PIL import Image
import imageio

# YUV 이미지 로드 함수
def load_yuv_images(folder_path, file_names, image_size=(128, 128)):
    images = []
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'rb') as file:
            image = np.frombuffer(file.read(), dtype=np.uint8)
            image = image.reshape(image_size)
            images.append(image)
    return np.array(images)

# 이미지 차이를 계산하는 함수
def process_images(images):
    images = images.astype(np.int16)  # 데이터 타입 변경
    return images - np.roll(images, -1, axis=0)

def create_histogram(images, title):
    plt.figure(figsize=(6, 6))
    
    # 모든 이미지 픽셀 값을 1차원 배열로 변환
    all_pixels = images.ravel()
    
    # 히스토그램 플롯
    counts, bins, patches = plt.hist(all_pixels, bins=100, color='gray', alpha=0.75)
    plt.title(f'{title} Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Count')  # Y축 레이블을 'Count'로 설정

    # Y축 눈금을 count로 설정
    plt.ylim(0, counts.max() * 1.1)  # Y축 범위를 최대값보다 약간 크게 설정
    
    # Y축 눈금 형식을 정수로 설정
    plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    img = Image.open(buf)
    img_array = np.array(img)

    return img_array

# 베이스 경로 설정
base_path = '/home/jijang/projects/Bacteria/dataset/B_cereus/240321'

# wandb 설정
wandb.init(project="B_cereus", name="240321")

# 클래스 디렉토리 리스트
classes = [str(i) for i in range(5)]

# 클래스별 step 초기화
for class_name in classes:
    class_path = os.path.join(base_path, class_name)
    video_folders = sorted([f for f in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, f))])

    if video_folders:
        video_folder = video_folders[0]  # 첫 번째 비디오 폴더 선택
        video_folder_path = os.path.join(class_path, video_folder)
        print(video_folder_path)
        yuv_files = sorted([f for f in os.listdir(video_folder_path) if f.endswith('.yuv')])

        if len(yuv_files) > 2:  # yuv 파일이 3개 이상인 경우
            yuv_files = yuv_files[1:-1]  # 첫 번째와 마지막 파일 제외
            images = load_yuv_images(video_folder_path, yuv_files)

            # 이미지 차이 계산
            processed_images = process_images(images)

            # 히스토그램 생성
            histogram_image = create_histogram(processed_images, f'Class {class_name} Video {video_folder}')
            wandb.log({f'{class_name}/histogram': wandb.Image(histogram_image, caption=f'Class {class_name} Histogram')})

            # 이미지 리스트 준비
            for step, image in enumerate(processed_images):
                wandb.log({f'{class_name}/residual_image': wandb.Image(image, caption=f'Image {step+1}')})

            # 비디오 생성 및 wandb에 로그
            video_path = os.path.join(video_folder_path, f'class_{class_name}_video.mp4')
            imageio.mimsave(video_path, processed_images, fps=5)  # 이미지 배열을 비디오로 저장
            wandb.log({f'{class_name}/residual_video': wandb.Video(video_path, caption=f'Class {class_name} Video')})

wandb.finish()
