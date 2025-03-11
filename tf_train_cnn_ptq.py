import argparse
import random
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from sklearn.metrics import confusion_matrix
from scipy.signal import find_peaks
from PIL import Image
import imageio
from skimage import exposure
import gc
from tensorflow.keras import backend as K

# GPU 설정 및 메모리 성장 활성화
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Enabled memory growth for {len(gpus)} GPU(s).")
    except RuntimeError as e:
        print(e)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


set_seed(42)


def create_model(input_shape, num_classes):
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


def process_images(structure, img_roi, num_classes, apply_equalization):
    X_data = []
    y_data = []

    for date_folder, subfolders in structure.items():
        for subfolder, video_folders in subfolders.items():
            for video_folder, segments in video_folders.items():
                for segment_id, file_paths in segments.items():
                    if len(file_paths) <= 1:
                        continue

                    X = get_images(file_paths, apply_equalization)

                    for _ in range(len(file_paths) - 1):
                        y_data.append(int(subfolder))  # subfolder를 정수 레이블로 사용

                    # 연속 프레임 간 차이 계산
                    X_processed = abs(X - np.roll(X, -1, axis=0))
                    X_processed = np.delete(X_processed, -1, axis=0)
                    X_data.append(X_processed)

    if not X_data:
        raise ValueError("No data found. Please check your dataset structure.")

    X_data = np.vstack(X_data).astype(np.float32)
    X_data = np.expand_dims(X_data, axis=-1)
    y_data = to_categorical(y_data, num_classes)
    X_data = X_data[:, :img_roi, :img_roi, :] 

    return X_data, y_data


def find_all_stable_segments(summed_diffs, peaks, buffer_size):
    total_frames = len(summed_diffs)
    excluded_indices = set()
    for peak in peaks:
        for i in range(max(0, peak - buffer_size), min(total_frames, peak + buffer_size + 1)):
            excluded_indices.add(i)
    valid_frames = [i for i in range(total_frames) if i not in excluded_indices]
    return valid_frames


def get_temporary_structure_with_stable_segments(base_path, date_folders, subfolders, stability_threshold, buffer_size):
    structure = {}
    for date_folder in date_folders:
        date_path = os.path.join(base_path, date_folder)
        if os.path.exists(date_path):
            structure[date_folder] = {}
            for subfolder in subfolders:
                subfolder_path = os.path.join(date_path, subfolder)
                if os.path.exists(subfolder_path):
                    structure[date_folder][subfolder] = {}
                    for video_name_folder in os.listdir(subfolder_path):
                        video_folder_path = os.path.join(subfolder_path, video_name_folder)
                        if os.path.isdir(video_folder_path):
                            video_files = [f for f in os.listdir(video_folder_path) if f.endswith('.yuv') or f.endswith('.tiff')]
                            video_files_sorted = sorted(video_files)
                            stable_segments = process_video_files(video_folder_path, video_files_sorted, stability_threshold, buffer_size)
                            if stable_segments:
                                structure[date_folder][subfolder][video_name_folder] = {}
                                start = stable_segments[0]
                                segment_id = 0
                                for i in range(1, len(stable_segments)):
                                    if stable_segments[i] != stable_segments[i - 1] + 1:
                                        end = stable_segments[i - 1] + 1
                                        segment_files = [os.path.join(video_folder_path, video_files_sorted[j]) for j in range(start, end)]
                                        structure[date_folder][subfolder][video_name_folder][f"segment_{segment_id}"] = segment_files
                                        segment_id += 1
                                        start = stable_segments[i]
                                end = stable_segments[-1] + 1
                                segment_files = [os.path.join(video_folder_path, video_files_sorted[j]) for j in range(start, end)]
                                structure[date_folder][subfolder][video_name_folder][f"segment_{segment_id}"] = segment_files
    return structure


def get_images(file_paths, apply_equalization):
    images = []
    for file_path in file_paths:
        if file_path.endswith('.tiff'):
            image = load_tiff_image(file_path, apply_equalization)
        else:
            image = load_yuv_image(file_path, apply_equalization)
        images.append(image)
    return np.array(images)


def load_tiff_image(file_path, apply_equalization):
    image = imageio.imread(file_path)
    
    # 16-bit 이미지인 경우 정규화
    if image.dtype == np.uint16:
        image = image.astype(np.float32) / 65535.0
    elif image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)
    
    # 히스토그램 평활화 적용 여부
    if apply_equalization:
        image = exposure.equalize_hist(image)
    
    return image


def load_yuv_image(file_path, apply_equalization=False):
    try:
        file_size = os.path.getsize(file_path)
        required_size = 128 * 128

        if file_size < required_size:
            raise ValueError(f"File '{file_path}' size ({file_size} bytes) is smaller than required size ({required_size} bytes).")

        with open(file_path, 'rb') as file:
            raw_data = file.read(required_size)

        if len(raw_data) != required_size:
            raise ValueError(f"Read data size ({len(raw_data)} bytes) from file '{file_path}' does not match required size ({required_size} bytes).")

        image = np.frombuffer(raw_data, dtype=np.uint8).reshape((128, 128))
        image = image.astype(np.float32) / 255.0

        if apply_equalization:
            image = exposure.equalize_hist(image)

        return image

    except Exception as e:
        print(f"Error loading YUV image from file '{file_path}': {e}")
        raise


def process_video_files(folder_path, file_names, stability_threshold, buffer_size):
    if not file_names:
        print("The folder is empty. No files to process.")
        return []
    if file_names[0].endswith('.yuv'):
        images = load_yuv_images(folder_path, file_names)
    elif file_names[0].endswith('.tiff'):
        images = load_tiff_images(folder_path, file_names)
    else:
        raise ValueError("Unsupported file format")
    image_diffs = np.abs(images[:-1] - images[1:])
    summed_diffs = image_diffs.sum(axis=(1, 2))
    peaks, _ = find_peaks(summed_diffs, height=stability_threshold)
    stable_segments = find_all_stable_segments(summed_diffs, peaks, buffer_size)
    return stable_segments


def load_tiff_images(folder_path, file_names):
    images = []
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        image = load_tiff_image(file_path, apply_equalization=False)
        images.append(image)
    return np.array(images)


def load_yuv_images(folder_path, file_names):
    images = []
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        image = load_yuv_image(file_path, apply_equalization=False)
        images.append(image)
    return np.array(images)


def main(args):
    os.makedirs(args.models_dir, exist_ok=True)
    excel_path = os.path.join(args.models_dir, "test_results.xlsx")
    df_dummy = pd.DataFrame()
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_dummy.to_excel(writer)

    # 날짜 폴더 정렬
    date_folders = sorted([d for d in os.listdir(args.base_path) if os.path.isdir(os.path.join(args.base_path, d))])
    print(f"Total date folders found: {date_folders}")

    if not date_folders:
        raise ValueError("No date folders found in the base path. Please check the dataset directory.")

    num_folds = len(date_folders)
    accuracy_results = []

    for fold_index, test_folder in enumerate(date_folders):
        print(f"Starting Fold {fold_index + 1} - Test Folder: {test_folder}")

        # 테스트 및 훈련 폴더 할당
        val_folder = [test_folder]
        train_folders = [folder for folder in date_folders if folder != test_folder]

        print(f"Train Folders: {train_folders}")
        print(f"Validation Folder: {val_folder}")

        # 훈련 및 검증 구조 준비
        train_structure = get_temporary_structure_with_stable_segments(
            args.base_path, train_folders, args.subfolders, args.stability_threshold, args.buffer_size
        )
        val_structure = get_temporary_structure_with_stable_segments(
            args.base_path, val_folder, args.subfolders, args.stability_threshold, args.buffer_size
        )

        # 이미지 처리
        try:
            X_train, y_train = process_images(train_structure, args.img_roi, args.num_classes, args.equalize)
            X_val, y_val = process_images(val_structure, args.img_roi, args.num_classes, args.equalize)
        except ValueError as e:
            print(f"Error processing images for Fold {fold_index + 1}: {e}")
            continue

        print(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")
        print(f"Validation data shape: {X_val.shape}, Validation labels shape: {y_val.shape}")

        # 모델 생성 및 컴파일 (PTQ를 위한 QAT 과정은 없음)
        model = create_model((args.img_roi, args.img_roi, 1), args.num_classes)
        optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        model.summary()

        # 모델 훈련
        model.fit(X_train, y_train, epochs=args.epochs, verbose=1, batch_size=64, validation_data=(X_val, y_val))

        # 모델 저장 (.h5 형식)
        model_save_path = os.path.join(args.models_dir, f"Fold_{fold_index + 1}_model.h5")
        model.save(model_save_path)

        # 저장된 모델 재로드 (메모리 해제 후 재로드)
        del model
        gc.collect()
        model = tf.keras.models.load_model(model_save_path)

        # 모델 평가
        val_loss, val_acc = model.evaluate(X_val, y_val)
        accuracy_results.append(val_acc)

        # 엑셀에 폴드 결과 저장
        fold_result = pd.DataFrame({
            'Fold': [fold_index + 1],
            'Test_Folder': [test_folder],
            'Accuracy': [val_acc]
        })
        with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a') as writer:
            fold_result.to_excel(writer, sheet_name=f'Fold_{fold_index + 1}', index=False)

        # 혼동 행렬 출력
        y_pred = model.predict(X_val)
        y_pred = np.argmax(y_pred, axis=1)
        y_test = np.argmax(y_val, axis=1)
        result = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix for Fold {fold_index + 1}:\n", result)

        # ============================
        # PTQ: TFLite 변환 (학습 후 양자화)
        # ============================
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        if args.quantization_mode == "fp16":
            print("Applying FP16 quantization...")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        elif args.quantization_mode == "int8":
            print("Applying INT8 quantization...")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            # 대표 데이터셋 제공: 학습 데이터 중 일부를 사용 (여기서는 X_train 사용)
            converter.representative_dataset = lambda: (
                [input_tensor] for input_tensor in tf.data.Dataset.from_tensor_slices(X_train).batch(1).take(100)
            )
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        elif args.quantization_mode == "fp8":
            raise ValueError("FP8 quantization is not supported by TFLite.")
        elif args.quantization_mode == "int4":
            raise ValueError("INT4 quantization is not supported by TFLite.")
        else:
            print("No quantization applied (full precision).")

        tflite_model = converter.convert()
        tflite_save_path = os.path.join(args.models_dir, f"Fold_{fold_index + 1}_model_{args.quantization_mode}.tflite")
        with open(tflite_save_path, 'wb') as f:
            f.write(tflite_model)

        # 메모리 정리
        K.clear_session()
        del model
        gc.collect()
        print(f"Completed Fold {fold_index + 1} and cleared memory.\n")

    # 최종 결과 요약
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

    with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a') as writer:
        final_results.to_excel(writer, sheet_name='Individual Folds', index=False)
        final_summary.to_excel(writer, sheet_name='Summary', index=False)

    print("Cross Validation Complete. Results saved to Excel.")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Script to train and evaluate a CNN with PTQ options (FP16, INT8, etc.) on image data.")
    parser.add_argument('--base_path', type=str, default='/home/jijang/ssd_data/projects/Bacteria/dataset/case_test/case16', help='Base directory for the dataset.')
    parser.add_argument('--models_dir', type=str, default='/home/jijang/ssd_data/projects/Bacteria/models/case_test/ptq_fp16_case16', help='Directory where models are saved.')
    parser.add_argument('--subfolders', type=str, nargs='+', default=['0', '1', '2', '3', '4'], help='Subfolders to include as classes.')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes to predict.')
    parser.add_argument('--img_frame', type=int, default=900, help='Frame size of the images.')
    parser.add_argument('--img_roi', type=int, default=96, help='Region of interest size for each image.')
    parser.add_argument('--stability_threshold', type=int, default=350, help='Threshold for stability in video segmentation.')
    parser.add_argument('--buffer_size', type=int, default=2, help='Buffer size around detected peaks in stability analysis.')
    parser.add_argument('--learning_rate', type=float, default=0.00001, help='Learning rate for the optimizer.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model.')
    parser.add_argument('--equalize', action='store_true', help='Apply histogram equalization to images.')
    parser.add_argument('--quantization_mode', type=str, default='fp16', choices=['none', 'fp16', 'int8'],
                        help='Quantization mode for PTQ')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
