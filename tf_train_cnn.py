import argparse
import random
import os
import gc

import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from sklearn.metrics import confusion_matrix
from scipy.signal import find_peaks 
import imageio
from skimage import exposure
from tensorflow.keras import backend as K
 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Enabled memory growth for {len(gpus)} GPU(s).")
    except RuntimeError as e:
        print(e)

# ---------------------------------------------------------
# 재현 가능성을 위해 랜덤 시드를 고정하는 함수
# ---------------------------------------------------------
def set_seed(seed: int) -> None:
    """
    랜덤 시드를 고정하여 실험 재현성을 높이기 위한 함수입니다.
    numpy, random, tensorflow 각각에 대해 seed를 설정해줍니다.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(42)

# ---------------------------------------------------------
# CNN 모델을 생성하는 함수
# ---------------------------------------------------------
def create_model(input_shape: tuple, num_classes: int) -> Sequential:
    """
    간단한 CNN 모델을 생성하여 반환합니다.
    - input_shape: 입력 이미지의 형태 (height, width, channel)
    - num_classes: 분류하려는 클래스 개수
    """
    model = Sequential([
        Conv2D(32, (5, 5), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (1, 1), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

# ---------------------------------------------------------
# 이미지 데이터셋(X)과 라벨(y)을 생성하는 함수
# 1) 구조화된 폴더(데이터) 정보를 바탕으로 이미지를 읽어옴
# 2) 순차 차분(프레임 간 차분)을 수행하여 X를 만든 뒤
# 3) 라벨(subfolder 정보)를 y로 구성
# ---------------------------------------------------------
def process_images(structure: dict, img_roi: int, num_classes: int, apply_equalization: bool):
    """
    구조화된 폴더(structure)에 들어있는 이미지 파일들을 로드하고,
    차분을 통해 CNN 학습용 X, y 데이터를 생성합니다.
    
    - structure: 안정 구간(움직임이 적은 구간) 별로 정리된 파일 경로 정보
    - img_roi: 이미지에서 사용할 ROI(Region of Interest) 크기 (ex. 96이면 96x96)
    - num_classes: 분류 클래스 개수
    - apply_equalization: 히스토그램 평활화 적용 여부
    
    반환:
      X_data: (N, img_roi, img_roi, 1) 형태의 차분 결과 이미지
      y_data: (N, num_classes) 형태의 One-hot 라벨
    """
    X_data = []
    y_data = []

    # structure 예시: { 날짜폴더: { subfolder(클래스): { 영상폴더: { segment_id: [파일경로들] } } } }
    for date_folder, subfolders in structure.items():
        for subfolder, video_folders in subfolders.items():
            for video_folder, segments in video_folders.items():
                for segment_id, file_paths in segments.items():
                    # 프레임이 1개 이하라면 차분할 수 없으므로 넘어감
                    if len(file_paths) <= 1:
                        continue

                    # TIFF 또는 YUV 이미지를 불러옴
                    X = get_images(file_paths, apply_equalization)

                    # 각 세그먼트 내 파일 개수 - 1 만큼 라벨 생성
                    # (예: 5장 이미지 -> 4개의 차분 결과 -> 4개의 라벨)
                    for _ in range(len(file_paths) - 1):
                        y_data.append(int(subfolder))  # subfolder 자체가 클래스이므로 int 변환

                    # 순차 차분: X - X(한 프레임 뒤)
                    # axis=0: 프레임 차원 기준, np.roll은 각 이미지들을 한 칸씩 '뒤로' 밀어줌
                    X_processed = abs(X - np.roll(X, -1, axis=0))
                    # 마지막 프레임은 차분이 불가능하므로 삭제
                    X_processed = np.delete(X_processed, -1, axis=0)
                    # 모든 차분 결과를 X_data 리스트에 추가
                    X_data.append(X_processed)

    # 실제로 수집된 데이터가 하나도 없다면 에러
    if not X_data:
        raise ValueError("No data found. Please check your dataset structure.")

    # 리스트에 쌓인 모든 차분 결과를 하나의 배열로 만듦
    X_data = np.vstack(X_data).astype(np.float32)
    
    # 채널 차원 추가 (Gray scale -> (N, H, W, 1))
    X_data = np.expand_dims(X_data, axis=-1)
    
    # 라벨을 One-hot 형태로 변환
    y_data = to_categorical(y_data, num_classes)
    
    # ROI 영역만큼 자르기 (이미지가 128x128이라면, 96x96만 사용)
    X_data = X_data[:, :img_roi, :img_roi, :]

    return X_data, y_data

# ---------------------------------------------------------
# 영상의 프레임별 차분 합에서 특정 피크(움직임이 큰 구간)를 찾고,
# 피크 주변을 제외한 '안정 구간' 인덱스를 리턴하는 함수
# ---------------------------------------------------------
def find_all_stable_segments(summed_diffs: np.ndarray, peaks: np.ndarray, buffer_size: int) -> list:
    """
    - summed_diffs: 각 프레임별로 sum(diff)를 계산한 값 (프레임 간 차분의 합)
    - peaks: scipy.signal.find_peaks를 통해 찾은 움직임이 큰 지점(피크)의 인덱스
    - buffer_size: 피크 주변을 배제할 범위 (피크를 중심으로 ±buffer_size)

    반환값: 움직임이 적은 구간(피크 주변을 제외한 프레임)의 인덱스 리스트
    """
    total_frames = len(summed_diffs)
    excluded_indices = set()
    
    # 모든 피크에 대해, 피크 ± buffer_size 범위의 인덱스를 제외 대상으로 등록
    for peak in peaks:
        for i in range(max(0, peak - buffer_size), min(total_frames, peak + buffer_size + 1)):
            excluded_indices.add(i)
    
    # 제외되지 않은 인덱스들만 모아서 반환
    valid_frames = [i for i in range(total_frames) if i not in excluded_indices]
    return valid_frames

# ---------------------------------------------------------
# (date_folder, subfolder, video_folder) 구조 내에서,
# 각 영상마다 '안정 구간'을 찾아 임시 구조체를 만드는 함수
# ---------------------------------------------------------
def get_temporary_structure_with_stable_segments(base_path: str,
                                                 date_folders: list,
                                                 subfolders: list,
                                                 stability_threshold: int,
                                                 buffer_size: int,
                                                 equalize: bool) -> dict:
    """
    각 date_folder와 subfolder(클래스) 내 영상 폴더를 탐색하며,
    영상 내에서 움직임이 작은 프레임들을 추출한 후,
    이를 세그먼트 단위로 묶어 structure(dict) 형태로 반환합니다.

    - base_path: 전체 데이터셋이 존재하는 최상위 폴더 경로
    - date_folders: 처리할 날짜 폴더 리스트
    - subfolders: 각 날짜 폴더 내 분류 클래스를 의미하는 서브폴더 리스트
    - stability_threshold: 영상 차분 합에서 피크를 찾을 때 사용할 임계값
    - buffer_size: 피크 주변(움직임이 큰 구간) 제외 범위
    - equalize: 히스토그램 평활화 적용 여부
    """
    structure = {}
    for date_folder in date_folders:
        date_path = os.path.join(base_path, date_folder)
        
        if os.path.exists(date_path):
            structure[date_folder] = {}
            
            for subfolder in subfolders:
                subfolder_path = os.path.join(date_path, subfolder)
                
                if os.path.exists(subfolder_path):
                    structure[date_folder][subfolder] = {}
                    
                    # subfolder 안에 있는 video_name_folder(각 영상)의 폴더 탐색
                    for video_name_folder in os.listdir(subfolder_path):
                        video_folder_path = os.path.join(subfolder_path, video_name_folder)
                        
                        # 실제 폴더인지 확인
                        if os.path.isdir(video_folder_path):
                            # 해당 폴더 내 모든 YUV/TIFF 파일을 찾는다
                            video_files = [
                                f for f in os.listdir(video_folder_path)
                                if f.endswith('.yuv') or f.endswith('.tiff')
                            ]
                            # 파일 이름 정렬
                            video_files_sorted = sorted(video_files)

                            # 영상 내에서 안정 구간(움직임 적은 구간)의 인덱스들을 찾음
                            stable_segments = process_video_files(
                                video_folder_path,
                                video_files_sorted,
                                stability_threshold,
                                buffer_size,
                                equalize
                            )
                            
                            if stable_segments:
                                structure[date_folder][subfolder][video_name_folder] = {}
                                start = stable_segments[0]
                                segment_id = 0

                                # 연속된 프레임끼리는 하나의 세그먼트로 묶음
                                for i in range(1, len(stable_segments)):
                                    # 만약 현재 프레임이 바로 이전 프레임+1이 아니라면,
                                    # 하나의 안정 구간이 마무리된 것으로 보고 새 세그먼트로 나눈다
                                    if stable_segments[i] != stable_segments[i - 1] + 1:
                                        end = stable_segments[i - 1] + 1
                                        segment_files = [
                                            os.path.join(video_folder_path, video_files_sorted[j])
                                            for j in range(start, end)
                                        ]
                                        structure[date_folder][subfolder][video_name_folder][
                                            f"segment_{segment_id}"] = segment_files
                                        segment_id += 1
                                        start = stable_segments[i]
                                
                                # 마지막 구간 처리
                                end = stable_segments[-1] + 1
                                segment_files = [
                                    os.path.join(video_folder_path, video_files_sorted[j])
                                    for j in range(start, end)
                                ]
                                structure[date_folder][subfolder][video_name_folder][
                                    f"segment_{segment_id}"] = segment_files

    return structure

# ---------------------------------------------------------
# 파일 경로 리스트를 받아 해당 TIFF 또는 YUV 이미지를 numpy 배열로 로드하는 함수
# ---------------------------------------------------------
def get_images(file_paths: list, apply_equalization: bool) -> np.ndarray:
    """
    - file_paths: 이미지 파일 경로들의 리스트
    - apply_equalization: 히스토그램 평활화 적용 여부
    
    TIFF/YUV 각각의 형식에 맞게 적절한 로더 함수를 호출하여
    (N, H, W) 형태의 배열로 만들어 반환합니다.
    """
    images = []
    for file_path in file_paths:
        if file_path.endswith('.tiff'):
            image = load_tiff_image(file_path, apply_equalization)
        else:
            image = load_yuv_image(file_path, apply_equalization)
        images.append(image)
    return np.array(images)

# ---------------------------------------------------------
# TIFF 파일을 열어 (0~1 범위로 스케일링) 후,
# 필요 시 히스토그램 평활화를 적용한 이미지를 반환
# ---------------------------------------------------------
def load_tiff_image(file_path: str, apply_equalization: bool) -> np.ndarray:
    """
    TIFF 이미지를 열어 float32(0~1 범위)로 변환하고,
    apply_equalization이 True라면 히스토그램 평활화를 적용 후 반환합니다.
    """
    # TIFF 파일 읽기 (16bit TIFF 가정)
    image = imageio.imread(file_path).astype(np.float32) / 65535.0
    
    # 히스토그램 평활화 옵션 적용
    if apply_equalization:
        image = exposure.equalize_hist(image)
    return image

# ---------------------------------------------------------
# 8-bit YUV(단일 채널, 128x128) 파일을 열어 0~1 범위로 변환
# 필요 시 히스토그램 평활화를 적용
# ---------------------------------------------------------
def load_yuv_image(file_path: str, apply_equalization: bool = False) -> np.ndarray:
    """
    YUV(8-bit, 128x128) 포맷의 파일 하나를 열어 Gray scale 이미지를 얻은 뒤,
    0~1 범위로 스케일링하고 필요 시 히스토그램 평활화를 적용하여 반환합니다.
    """
    try:
        file_size = os.path.getsize(file_path)
        required_size = 128 * 128  # 128x128 픽셀
        
        if file_size < required_size:
            raise ValueError(f"File '{file_path}' size ({file_size} bytes) "
                             f"is smaller than required size ({required_size} bytes).")

        with open(file_path, 'rb') as file:
            raw_data = file.read(required_size)

        if len(raw_data) != required_size:
            raise ValueError(f"Read data size ({len(raw_data)} bytes) from file '{file_path}' "
                             f"does not match required size ({required_size} bytes).")

        # 8-bit 데이터(0~255)를 (128,128)로 reshape
        image = np.frombuffer(raw_data, dtype=np.uint8).reshape((128, 128))
        image = image.astype(np.float32) / 255.0

        # 히스토그램 평활화 적용 여부
        if apply_equalization:
            image = exposure.equalize_hist(image)

        return image

    except Exception as e:
        print(f"Error loading YUV image from file '{file_path}': {e}")
        raise

# ---------------------------------------------------------
# 한 폴더 내의 TIFF 또는 YUV 파일들을 순서대로 로드하여
# 각 프레임 사이의 차분을 구하고, 그 합(summed_diffs)을 구하여
# 움직임이 큰 지점(피크)을 찾고, 안정 구간 인덱스를 반환
# ---------------------------------------------------------
def process_video_files(folder_path: str,
                        file_names: list,
                        stability_threshold: int,
                        buffer_size: int,
                        equalize: bool) -> list:
    """
    - folder_path: TIFF/YUV 파일이 있는 폴더 경로
    - file_names: 폴더 내 파일 이름들 (정렬된 상태)
    - stability_threshold: find_peaks에서 사용할 임계값
    - buffer_size: 피크 주변 배제 범위
    - equalize: 히스토그램 평활화 여부

    1) TIFF/YUV 파일을 한 번에 로드
    2) 인접 프레임 차분 결과를 합산
    3) 크기가 stability_threshold 이상인 피크를 찾음
    4) 피크 주변(±buffer_size)을 제외한 인덱스를 '안정 구간'으로 간주하고 반환
    """
    if not file_names:
        print("The folder is empty. No files to process.")
        return []

    # 첫 파일 확장자를 보고 TIFF 또는 YUV인지 결정
    if file_names[0].endswith('.yuv'):
        images = load_yuv_images(folder_path, file_names)
    elif file_names[0].endswith('.tiff'):
        images = load_tiff_images(folder_path, file_names, equalize)
    else:
        raise ValueError("Unsupported file format")

    # 프레임 간 차분
    image_diffs = np.abs(images[:-1] - images[1:])
    # 차분 결과를 프레임 단위로 합산
    summed_diffs = image_diffs.sum(axis=(1, 2))

    # SciPy의 find_peaks로 이동이 큰 지점(피크)을 찾음
    # height=stability_threshold 이상인 지점만 피크로 인정
    peaks, _ = find_peaks(summed_diffs, height=stability_threshold)

    # 피크 주변을 제외한 안정 구간 인덱스 리스트 추출
    stable_segments = find_all_stable_segments(summed_diffs, peaks, buffer_size)
    return stable_segments

# ---------------------------------------------------------
# 여러 TIFF 이미지를 한 번에 읽어 (N, 128, 128) 배열로 반환
# ---------------------------------------------------------
def load_tiff_images(folder_path: str, file_names: list, equalize: bool) -> np.ndarray:
    """
    여러 TIFF 파일을 순회하며 load_tiff_image를 이용해 로드하고,
    리스트에 쌓아 numpy 배열로 반환합니다.
    """
    images = []
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        image = load_tiff_image(file_path, equalize)
        images.append(image)
    return np.array(images)

# ---------------------------------------------------------
# 여러 YUV 이미지를 한 번에 읽어 (N, 128, 128) 배열로 반환
# ---------------------------------------------------------
def load_yuv_images(folder_path: str, file_names: list) -> np.ndarray:
    """
    여러 YUV 파일을 순회하며 load_yuv_image를 이용해 로드하고,
    리스트에 쌓아 numpy 배열로 반환합니다.
    """
    images = []
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        image = load_yuv_image(file_path, apply_equalization=False)
        images.append(image)
    return np.array(images)

# ---------------------------------------------------------
# 메인 함수
# 1) 날짜 폴더 리스트(date_folders)를 읽고 Fold를 결정
# 2) K-Fold Cross Validation 방식으로:
#    - test_folder(= validation folder)를 지정
#    - 나머지 folder를 train_folders로 사용
# 3) 구조 체계를 만들고(process_video_files -> get_temporary_structure_with_stable_segments),
#    process_images로 X, y 데이터를 만든 뒤, 모델 학습/평가
# 4) 결과(혼동행렬, 정확도)를 Excel에 저장
# ---------------------------------------------------------
def main(args):
    """
    메인 실행 함수로, 인자로 받은 args를 활용하여
    K-Fold Cross Validation을 수행하고 모델 및 결과를 저장합니다.
    """
    # 모델 저장 경로 생성
    os.makedirs(args.models_dir, exist_ok=True)
    excel_path = os.path.join(args.models_dir, "test_results.xlsx")

    # 추후 엑셀에 결과를 기록하기 위해 초기화
    df_dummy = pd.DataFrame()
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_dummy.to_excel(writer)

    # 날짜 폴더(데이터 폴더)를 읽고 정렬
    date_folders = sorted([
        d for d in os.listdir(args.base_path)
        if os.path.isdir(os.path.join(args.base_path, d))
    ])
    print(f"Total date folders found: {date_folders}")

    if not date_folders:
        raise ValueError("No date folders found in the base path. Please check the dataset directory.")

    # K-Fold 개수는 날짜 폴더 수와 동일하게 설정 (1 folder = 1 fold)
    num_folds = len(date_folders)
    accuracy_results = []  # 각 폴드별 accuracy를 저장

    # -----------------------------------------------------
    # K-Fold Cross Validation 루프
    # -----------------------------------------------------
    for fold_index, test_folder in enumerate(date_folders):
        print(f"Starting Fold {fold_index + 1} - Test Folder: {test_folder}")

        # 해당 fold에서 test_folder(= validation folder)는 1개,
        # 나머지 모두 train_folders로 사용
        val_folder = [test_folder]
        train_folders = [folder for folder in date_folders if folder != test_folder]

        print(f"Train Folders: {train_folders}")
        print(f"Validation Folder: {val_folder}")

        # 안정 구간을 추출하여 구조화된 임시 구조 생성
        train_structure = get_temporary_structure_with_stable_segments(
            args.base_path, train_folders, args.subfolders,
            args.stability_threshold, args.buffer_size, args.equalize
        )
        val_structure = get_temporary_structure_with_stable_segments(
            args.base_path, val_folder, args.subfolders,
            args.stability_threshold, args.buffer_size, args.equalize
        )

        # 이미지와 라벨 생성
        try:
            X_train, y_train = process_images(train_structure, args.img_roi, args.num_classes, args.equalize)
            X_val, y_val = process_images(val_structure, args.img_roi, args.num_classes, args.equalize)
        except ValueError as e:
            print(f"Error processing images for Fold {fold_index + 1}: {e}")
            continue

        print(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")
        print(f"Validation data shape: {X_val.shape}, Validation labels shape: {y_val.shape}")

        # 모델 생성
        model = create_model((args.img_roi, args.img_roi, 1), args.num_classes)
        optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        model.summary()

        # 모델 학습
        model.fit(
            X_train, y_train,
            epochs=args.epochs,
            verbose=1,
            batch_size=64,
            validation_data=(X_val, y_val)
        )

        # 모델 저장
        model_save_path = os.path.join(args.models_dir, f"Fold_{fold_index + 1}_model.h5")
        model.save(model_save_path)

        # 메모리 정리
        del model
        gc.collect()

        # 모델 재로드
        model = tf.keras.models.load_model(model_save_path)

        # 모델 평가
        val_loss, val_acc = model.evaluate(X_val, y_val)
        accuracy_results.append(val_acc)

        # Fold 결과를 엑셀에 저장
        fold_result = pd.DataFrame({
            'Fold': [fold_index + 1],
            'Test_Folder': [test_folder],
            'Accuracy': [val_acc]
        })
        with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a') as writer:
            fold_result.to_excel(writer, sheet_name=f'Fold_{fold_index + 1}', index=False)

        # 혼동 행렬 계산
        y_pred = model.predict(X_val)
        y_pred = np.argmax(y_pred, axis=1)
        y_test = np.argmax(y_val, axis=1)
        result = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix for Fold {fold_index + 1}:\n", result)

        # TFLite 변환 및 저장
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        tflite_save_path = os.path.join(args.models_dir, f"Fold_{fold_index + 1}_model.tflite")
        with open(tflite_save_path, 'wb') as f:
            f.write(tflite_model)

        # 세션 클리어 후 메모리 해제
        K.clear_session()
        del model
        gc.collect()
        print(f"Completed Fold {fold_index + 1} and cleared memory.\n")

    # -----------------------------------------------------
    # 모든 폴드가 끝난 뒤, 정확도 요약 및 Excel 저장
    # -----------------------------------------------------
    final_results = pd.DataFrame({
        'Fold': range(1, num_folds + 1),
        'Test_Folder': date_folders,
        'Accuracy': accuracy_results
    })
    mean_accuracy = np.mean(accuracy_results)
    std_accuracy = np.std(accuracy_results)
    final_summary = pd.DataFrame({
        'Metric': ['Mean Accuracy', 'Standard Deviation'],
        'Value': [mean_accuracy, std_accuracy]
    })

    # 종합 결과를 Excel에 추가 저장
    with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a') as writer:
        final_results.to_excel(writer, sheet_name='Individual Folds', index=False)
        final_summary.to_excel(writer, sheet_name='Summary', index=False)

    print("Cross Validation Complete. Results saved to Excel.")

# ---------------------------------------------------------
# 명령줄 인수 파싱 함수
# ---------------------------------------------------------
def parse_arguments():
    """
    명령줄 인자를 파싱하여 Namespace 형태로 반환합니다.
    """
    parser = argparse.ArgumentParser(
        description="Script to train and evaluate a convolutional neural network on image data."
    )
    parser.add_argument('--base_path', type=str,
                        default='/home/jijang/ssd_data/projects/Bacteria/dataset/20231221_DS/Stationary',
                        help='Base directory for the dataset.')
    parser.add_argument('--models_dir', type=str,
                        default='/home/jijang/ssd_data/projects/Bacteria/models/20231221_DS/Stationary',
                        help='Directory where models are saved.')
    parser.add_argument('--subfolders', type=str, nargs='+', default=['0', '1', '2', '3'],
                        help='Subfolders to include as classes.')
    parser.add_argument('--num_classes', type=int, default=4,
                        help='Number of classes to predict.')
    parser.add_argument('--img_frame', type=int, default=300,
                        help='Frame size of the images (현재 사용되지 않는 인자).')
    parser.add_argument('--img_roi', type=int, default=96,
                        help='Region of interest size for each image.')
    parser.add_argument('--stability_threshold', type=int, default=350,
                        help='Threshold for stability in video segmentation.')
    parser.add_argument('--buffer_size', type=int, default=2,
                        help='Buffer size around detected peaks in stability analysis.')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train the model.')
    parser.add_argument('--equalize', action='store_true',
                        help='Apply histogram equalization to images.')
    return parser.parse_args()

# ---------------------------------------------------------
# 실제로 스크립트를 실행했을 때 동작할 메인 엔트리 포인트
# ---------------------------------------------------------
if __name__ == "__main__":
    args = parse_arguments()
    main(args)
