import os
import numpy as np
import matplotlib.pyplot as plt
import wandb
import io
import imageio
from PIL import Image
from skimage import exposure

# TIFF 이미지 로드 함수
def load_tiff_images(folder_path, file_names):
    images = []
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        image = imageio.imread(file_path)  # TIFF 파일 읽기
        images.append(image)
    return np.array(images)

# 이미지 차이를 계산하는 함수
def process_images(images):
    images = images.astype(np.int32)  # 데이터 타입 변경
    return images - np.roll(images, -1, axis=0)
 
def create_histogram(images, title):
    plt.figure(figsize=(6, 6))
    all_pixels = images.ravel()
    counts, bins, patches = plt.hist(all_pixels, bins=100, color='gray', alpha=0.75)
    plt.title(f'{title} Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Count')
    plt.ylim(0, counts.max() * 1.1)
    plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img = Image.open(buf)
    img_array = np.array(img)
    return img_array

# WandB 설정
wandb.init(project="Razer", name="20240611")

# 베이스 경로 설정
base_path = '/home/jijang/projects/Bacteria/dataset/Razer/20240611'
classes = [str(i) for i in range(5)]  # 클래스 디렉토리 리스트

# 클래스별 처리
for class_name in classes:
    class_path = os.path.join(base_path, class_name)
    video_folders = sorted([f for f in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, f))])

    if video_folders:
        video_folder = video_folders[0]
        video_folder_path = os.path.join(class_path, video_folder)
        tiff_files = sorted([f for f in os.listdir(video_folder_path) if f.endswith('.tiff')])

        if len(tiff_files) > 2:
            tiff_files = tiff_files[1:-1]
            images = load_tiff_images(video_folder_path, tiff_files)
            processed_images = process_images(images)

            # 원본 이미지의 비디오 및 히스토그램 생성
            original_video_path = os.path.join(video_folder_path, f'original_{class_name}_video.mp4')
            imageio.mimsave(original_video_path, processed_images, fps=5)
            wandb.log({f'{class_name}/original_video': wandb.Video(original_video_path, caption=f'Original {class_name} Video')})

            original_histogram_image = create_histogram(processed_images, f'Original {class_name} Histogram')
            wandb.log({f'{class_name}/original_histogram': wandb.Image(original_histogram_image, caption=f'Original {class_name} Histogram')})

            # 평활화된 이미지의 비디오 및 히스토그램 생성
            equalized_images = np.array([exposure.equalize_hist(img) for img in processed_images.astype(np.uint16)])
            equalized_video_path = os.path.join(video_folder_path, f'equalized_{class_name}_video.mp4')
            imageio.mimsave(equalized_video_path, equalized_images, fps=5)
            wandb.log({f'{class_name}/equalized_video': wandb.Video(equalized_video_path, caption=f'Equalized {class_name} Video')})

            equalized_histogram_image = create_histogram(equalized_images, f'Equalized {class_name} Histogram')
            wandb.log({f'{class_name}/equalized_histogram': wandb.Image(equalized_histogram_image, caption=f'Equalized {class_name} Histogram')})

wandb.finish()
