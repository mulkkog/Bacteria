import argparse
import random

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.signal import find_peaks
from PIL import Image
import imageio
from skimage import exposure
import gc
from tensorflow.keras import backend as K

# Spektral 관련 임포트
# !pip install spektral==1.0.2
from spektral.layers import GCNConv, GlobalAvgPool
from spektral.data import Dataset, Graph
from spektral.data.loaders import BatchLoader

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

def create_gcn_model(num_patches, num_features, num_classes):
    """
    GCN 모델을 생성합니다.
    """
    x_input = keras.Input(shape=(num_patches, num_features), name='X_input')
    a_input = keras.Input(shape=(num_patches, num_patches), sparse=True, name='A_input')
    
    x = GCNConv(32, activation='relu')([x_input, a_input])
    x = GCNConv(64, activation='relu')([x, a_input])
    x = GlobalAvgPool()(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=[x_input, a_input], outputs=outputs)
    return model

def image_to_patches(image, patch_size):
    """
    이미지를 patch_size x patch_size 조각으로 분할하여 배열 형태로 반환합니다.
    """
    patches = []
    for i in range(0, image.shape[0], patch_size):
        for j in range(0, image.shape[1], patch_size):
            patch = image[i:i+patch_size, j:j+patch_size]
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                patches.append(patch)
    return np.array(patches)

def create_grid_adjacency(patches_per_dim_height, patches_per_dim_width):
    """
    격자 그래프(4방향 연결)의 인접 행렬을 생성합니다.
    """
    num_nodes = patches_per_dim_height * patches_per_dim_width
    adj = np.zeros((num_nodes, num_nodes))
    for i in range(patches_per_dim_height):
        for j in range(patches_per_dim_width):
            node = i * patches_per_dim_width + j
            # 상하좌우 인접 노드 연결
            if i > 0:
                adj[node, (i - 1) * patches_per_dim_width + j] = 1
            if i < patches_per_dim_height - 1:
                adj[node, (i + 1) * patches_per_dim_width + j] = 1
            if j > 0:
                adj[node, i * patches_per_dim_width + (j - 1)] = 1
            if j < patches_per_dim_width - 1:
                adj[node, i * patches_per_dim_width + (j + 1)] = 1
    return adj

def get_images(file_paths, apply_equalization):
    """
    TIFF 또는 YUV 파일을 로드하고, 옵션에 따라 히스토그램 평활화를 적용한 뒤 넘파이 배열로 반환합니다.
    """
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
    
    # 16-bit 이미지 처리
    if image.dtype == np.uint16:
        image = image.astype(np.float32) / 65535.0
    elif image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)
    
    # 히스토그램 평활화
    if apply_equalization:
        image = exposure.equalize_hist(image)
    
    return image

def load_yuv_image(file_path, apply_equalization=False):
    """
    YUV(단일 채널) 파일을 128x128 크기로 읽어서 float32 (0~1) 스케일로 반환합니다.
    """
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

def process_images_to_graph(structure, num_classes, apply_equalization, patch_size=32):
    """
    구성된 structure(날짜/폴더/영상/세그먼트)에서 이미지를 읽어와,
    각 이미지 프레임마다 (X, A) 그래프, 그리고 클래스 레이블을 구성합니다.
    """
    graphs = []
    labels = []
    
    for date_folder, subfolders in structure.items():
        for subfolder, video_folders in subfolders.items():
            for video_folder, segments in video_folders.items():
                for segment_id, file_paths in segments.items():
                    if len(file_paths) <= 1:
                        continue

                    images = get_images(file_paths, apply_equalization)
                    
                    for image in images:
                        # 이미지 패치로 분할
                        patches = image_to_patches(image, patch_size)
                        num_patches = patches.shape[0]
                        
                        # 노드 특성: 각 패치를 평탄화
                        X = patches.reshape(num_patches, -1)
                        
                        # (128,128)이 전제된 경우, patches_per_dim = 128 // patch_size
                        patches_per_dim = image.shape[0] // patch_size
                        
                        # 인접 행렬 생성 (격자형 그래프)
                        A = create_grid_adjacency(patches_per_dim, patches_per_dim)

                        graphs.append((X, A))
                        labels.append(int(subfolder))  # subfolder가 '0' or '1'이면 int 변환
    
    return graphs, np.array(labels)

class CustomDataset(Dataset):
    """
    Spektral용 커스텀 Dataset.
    """
    def __init__(self, graphs, labels, **kwargs):
        self.graphs = graphs
        self.labels = labels
        super().__init__(**kwargs)

    def read(self):
        return [
            Graph(x=graph[0], a=graph[1], y=self.labels[idx]) 
            for idx, graph in enumerate(self.graphs)
        ]

def get_temporary_structure_with_stable_segments(base_path, date_folders, subfolders, stability_threshold, buffer_size):
    """
    안정적인 프레임 구간(segmentation)을 찾아 structure를 구성합니다.
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

def process_video_files(folder_path, file_names, stability_threshold, buffer_size):
    """
    해당 폴더 내 영상 파일들을 읽어서 프레임 차이를 계산하고,
    안정 구간(peak 주변을 제외한 구간)을 찾아 인덱스를 반환합니다.
    """
    if not file_names:
        print("The folder is empty. No files to process.")
        return []
    
    # 파일 포맷에 따라 로드
    if file_names[0].endswith('.yuv'):
        images = load_yuv_images(folder_path, file_names)
    elif file_names[0].endswith('.tiff'):
        images = load_tiff_images(folder_path, file_names)
    else:
        raise ValueError("Unsupported file format")
    
    # 인접 프레임 차이
    image_diffs = np.abs(images[:-1] - images[1:])
    summed_diffs = image_diffs.sum(axis=(1, 2))
    
    # 피크(불안정 구간) 찾기
    peaks, _ = find_peaks(summed_diffs, height=stability_threshold)
    
    # 안정 프레임 인덱스
    stable_segments = find_all_stable_segments(summed_diffs, peaks, buffer_size)
    return stable_segments

def find_all_stable_segments(summed_diffs, peaks, buffer_size):
    """
    높은 차이가 발생하는 peak 주변(buffer_size)을 제외한 구간을 안정 구간으로 간주.
    """
    total_frames = len(summed_diffs)
    excluded_indices = set()
    for peak in peaks:
        for i in range(max(0, peak - buffer_size), min(total_frames, peak + buffer_size + 1)):
            excluded_indices.add(i)
    valid_frames = [i for i in range(total_frames) if i not in excluded_indices]
    return valid_frames

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
    # 결과 저장 디렉토리 생성
    os.makedirs(args.models_dir, exist_ok=True)
    excel_path = os.path.join(args.models_dir, "test_results.xlsx")
    df_dummy = pd.DataFrame()
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_dummy.to_excel(writer)

    # 날짜 폴더 목록
    date_folders = [d for d in os.listdir(args.base_path) if os.path.isdir(os.path.join(args.base_path, d))]
    date_folders = sorted(date_folders)
    print(f"Total date folders found: {date_folders}")

    if not date_folders:
        raise ValueError("No date folders found in the base path.")

    # 직접 K-Fold 방식: 한 폴더씩 Validation, 나머지는 Train
    num_folds = len(date_folders)
    accuracy_results = []

    for fold_index, test_folder in enumerate(date_folders):
        print(f"Starting Fold {fold_index + 1} - Test Folder: {test_folder}")

        # Test(Val)로 사용할 폴더 & Train으로 사용할 폴더 구분
        val_folder = [test_folder]
        train_folders = [folder for folder in date_folders if folder != test_folder]

        print(f"Train Folders: {train_folders}")
        print(f"Validation Folder: {val_folder}")

        # 데이터 구조 생성 (train/val)
        train_structure = get_temporary_structure_with_stable_segments(
            args.base_path, train_folders, args.subfolders, args.stability_threshold, args.buffer_size
        )
        val_structure = get_temporary_structure_with_stable_segments(
            args.base_path, val_folder, args.subfolders, args.stability_threshold, args.buffer_size
        )

        # 그래프 데이터 준비
        X_train, y_train = process_images_to_graph(train_structure, args.num_classes, args.equalize, patch_size=32)
        X_val, y_val = process_images_to_graph(val_structure, args.num_classes, args.equalize, patch_size=32)

        # Spektral Dataset 및 BatchLoader
        train_dataset = CustomDataset(X_train, y_train)
        val_dataset = CustomDataset(X_val, y_val)

        loader_train = BatchLoader(train_dataset, batch_size=256, shuffle=True)
        loader_val = BatchLoader(val_dataset, batch_size=256, shuffle=False)

        # 모델 생성
        num_patches = X_train[0][0].shape[0]   # 예: 16
        num_features = X_train[0][0].shape[1]  # 예: patch_size*patch_size = 1024
        model = create_gcn_model(num_patches, num_features, args.num_classes)
        optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)
        model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.summary()

        # 모델 학습
        model.fit(
            loader_train.load(),
            steps_per_epoch=loader_train.steps_per_epoch,
            epochs=args.epochs,
            validation_data=loader_val.load(),
            validation_steps=loader_val.steps_per_epoch
        )

        # 모델 저장
        model_save_path = os.path.join(args.models_dir, f"Fold_{fold_index + 1}_model.h5")
        model.save(model_save_path, save_format='h5')

        # 모델 재로드 시 Spektral 커스텀 객체 등록 필요
        from spektral.layers import GCNConv, GlobalAvgPool
        model = keras.models.load_model(
            model_save_path, 
            compile=False,
            custom_objects={'GCNConv': GCNConv, 'GlobalAvgPool': GlobalAvgPool}
        )
        # 재컴파일
        model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        # 모델 평가
        val_loss, val_acc = model.evaluate(loader_val.load(), steps=loader_val.steps_per_epoch)
        accuracy_results.append(val_acc)

        # 엑셀에 결과 저장
        fold_result = pd.DataFrame({
            'Fold': [fold_index + 1],
            'Test_Folder': [test_folder],
            'Accuracy': [val_acc]
        })
        with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a') as writer:
            fold_result.to_excel(writer, sheet_name=f'Fold_{fold_index + 1}', index=False)

        # 혼동 행렬 계산
        y_pred = []
        y_true = []
        for _ in range(loader_val.steps_per_epoch):
            batch = next(loader_val.load())
            (x_batch, a_batch), labels_batch = batch  # Spektral BatchLoader가 ( (x, a), labels ) 형태로 반환
            preds = model.predict([x_batch, a_batch], verbose=0)
            y_pred.extend(np.argmax(preds, axis=1))
            y_true.extend(labels_batch)

        cm = confusion_matrix(y_true, y_pred)
        print(f"Confusion Matrix for Fold {fold_index + 1}:\n{cm}")

        # TFLite 변환
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            # 복잡한 TF ops를 TFLite에서 사용하기 위해
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,  
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            # tensor list ops 비활성화
            converter._experimental_lower_tensor_list_ops = False
            tflite_model = converter.convert()

            tflite_save_path = os.path.join(args.models_dir, f"Fold_{fold_index + 1}_model.tflite")
            with open(tflite_save_path, 'wb') as f:
                f.write(tflite_model)
            print(f"TFLite model saved at: {tflite_save_path}")
        except Exception as e:
            print(f"TFLite 변환 중 오류가 발생했습니다: {e}")

        # 메모리 정리
        K.clear_session()
        del model
        gc.collect()
        print(f"Completed Fold {fold_index + 1} and cleared memory.\n")

    # 최종 결과 정리
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
    parser = argparse.ArgumentParser(description="Script to train and evaluate a Graph Convolutional Network on image data.")
    parser.add_argument('--base_path', type=str, default='/home/jijang/ssd_data/projects/Bacteria/dataset/stable',
                        help='Base directory for the dataset.')
    parser.add_argument('--models_dir', type=str, default='/home/jijang/ssd_data/projects/Bacteria/models/stable_gnn',
                        help='Directory where models are saved.')
    parser.add_argument('--subfolders', type=str, nargs='+', default=['0', '1'],
                        help='Subfolders to include as classes.')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes to predict.')
    parser.add_argument('--img_frame', type=int, default=300, help='Frame size of the images.')
    parser.add_argument('--patch_size', type=int, default=32, help='Patch size to divide images for graph construction.')
    parser.add_argument('--stability_threshold', type=int, default=350, help='Threshold for stability in video segmentation.')
    parser.add_argument('--buffer_size', type=int, default=2, help='Buffer size around detected peaks in stability analysis.')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the optimizer.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model.')
    parser.add_argument('--equalize', action='store_true', help='Apply histogram equalization to images.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)