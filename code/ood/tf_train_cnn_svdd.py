import argparse
import random
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report

import imageio
from skimage import exposure
from scipy.signal import find_peaks
import pandas as pd
import gc

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Enabled memory growth for {len(gpus)} GPU(s).")
    except RuntimeError as e:
        print(e)

set_seed(42)


# ---------------------------
# 영상→세그먼트 구조 관련 함수들
# ---------------------------
def find_all_stable_segments(summed_diffs, peaks, buffer_size):
    total_frames = len(summed_diffs)
    excluded_indices = set()
    for peak in peaks:
        for i in range(max(0, peak - buffer_size), min(total_frames, peak + buffer_size + 1)):
            excluded_indices.add(i)
    valid_frames = [i for i in range(total_frames) if i not in excluded_indices]
    return valid_frames

def process_video_files(folder_path, file_names, stability_threshold, buffer_size):
    if not file_names:
        return []
    if file_names[0].endswith('.yuv'):
        images = load_yuv_images(folder_path, file_names)
    elif file_names[0].endswith('.tiff'):
        images = load_tiff_images(folder_path, file_names)
    else:
        raise ValueError("Unsupported file format")

    image_diffs = np.abs(images[:-1] - images[1:])
    summed_diffs = image_diffs.sum(axis=(1,2))
    peaks, _ = find_peaks(summed_diffs, height=stability_threshold)
    stable_segments = find_all_stable_segments(summed_diffs, peaks, buffer_size)
    return stable_segments

def get_temporary_structure_with_stable_segments(base_path, date_folders, subfolders, stability_threshold, buffer_size):
    structure = {}
    for date_folder in date_folders:
        date_path = os.path.join(base_path, date_folder)
        if not os.path.exists(date_path):
            continue
        structure[date_folder] = {}
        for subfolder in subfolders:
            sub_path = os.path.join(date_path, subfolder)
            if not os.path.exists(sub_path):
                continue
            structure[date_folder][subfolder] = {}
            for video_name_folder in os.listdir(sub_path):
                video_folder_path = os.path.join(sub_path, video_name_folder)
                if not os.path.isdir(video_folder_path):
                    continue
                video_files = sorted([f for f in os.listdir(video_folder_path) 
                                      if f.endswith('.yuv') or f.endswith('.tiff')])
                stable_segments = process_video_files(
                    video_folder_path, video_files, stability_threshold, buffer_size
                )
                if stable_segments:
                    structure[date_folder][subfolder][video_name_folder] = {}
                    start = stable_segments[0]
                    segment_id = 0
                    for i in range(1, len(stable_segments)):
                        if stable_segments[i] != stable_segments[i-1] + 1:
                            end = stable_segments[i-1] + 1
                            seg_files = [os.path.join(video_folder_path, video_files[j]) 
                                         for j in range(start, end)]
                            structure[date_folder][subfolder][video_name_folder][f"segment_{segment_id}"] = seg_files
                            segment_id += 1
                            start = stable_segments[i]
                    # 마지막 구간
                    end = stable_segments[-1] + 1
                    seg_files = [os.path.join(video_folder_path, video_files[j]) 
                                 for j in range(start, end)]
                    structure[date_folder][subfolder][video_name_folder][f"segment_{segment_id}"] = seg_files
    return structure


# ---------------------------
# 이미지 로드 함수
# ---------------------------
def load_yuv_image(file_path, apply_equalization=False):
    try:
        required_size = 128 * 128
        file_size = os.path.getsize(file_path)
        if file_size < required_size:
            raise ValueError(f"File size {file_size} < required {required_size}.")

        with open(file_path, 'rb') as f:
            raw_data = f.read(required_size)
        if len(raw_data) != required_size:
            raise ValueError(f"Read data {len(raw_data)} != {required_size}.")

        image = np.frombuffer(raw_data, dtype=np.uint8).reshape((128,128))
        image = image.astype(np.float32)/255.

        if apply_equalization:
            image = exposure.equalize_hist(image)
        return image
    except Exception as e:
        print(f"Error loading YUV: {e}")
        raise

def load_tiff_image(file_path, apply_equalization=False):
    image = imageio.imread(file_path)
    if image.dtype == np.uint16:
        image = image.astype(np.float32)/65535.0
    elif image.dtype == np.uint8:
        image = image.astype(np.float32)/255.0
    else:
        image = image.astype(np.float32)

    if apply_equalization:
        image = exposure.equalize_hist(image)
    return image

def load_yuv_images(folder_path, file_names):
    images = []
    for fn in file_names:
        fp = os.path.join(folder_path, fn)
        images.append(load_yuv_image(fp, apply_equalization=False))
    return np.array(images)

def load_tiff_images(folder_path, file_names):
    images = []
    for fn in file_names:
        fp = os.path.join(folder_path, fn)
        images.append(load_tiff_image(fp, apply_equalization=False))
    return np.array(images)

def get_images(file_paths, apply_equalization):
    frames = []
    for fp in file_paths:
        if fp.endswith('.tiff'):
            img = load_tiff_image(fp, apply_equalization)
        else:
            img = load_yuv_image(fp, apply_equalization)
        frames.append(img)
    return np.array(frames)


# ---------------------------
# One-Class용 데이터 처리
# ---------------------------
def process_images(structure, img_roi, apply_equalization):
    """
    구조 내의 세그먼트 frame들을 로드 → frame 차이 → 스택
    반환 shape: (N, img_roi, img_roi, 1)
    """
    X_data = []
    for date_folder, subfolders in structure.items():
        for subfolder, video_dict in subfolders.items():
            for video_id, segments in video_dict.items():
                for seg_id, file_paths in segments.items():
                    if len(file_paths) <= 1:
                        continue
                    # 프레임 로드
                    X_frames = get_images(file_paths, apply_equalization)
                    # frame diff
                    X_diff = np.abs(X_frames - np.roll(X_frames, -1, axis=0))
                    X_diff = X_diff[:-1]  # 마지막 하나 제거
                    X_data.append(X_diff)
    if not X_data:
        raise ValueError("No data found for process_images.")
    X_data = np.vstack(X_data).astype(np.float32)
    X_data = np.expand_dims(X_data, axis=-1)  # channel
    # roi crop
    X_data = X_data[:,:img_roi,:img_roi,:]
    return X_data


# ---------------------------
# One-Class(Deep SVDD 스타일) 핵심
# ---------------------------
def create_one_class_model(input_shape, embedding_dim=32):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, (5,5), activation='relu')(inputs)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, (1,1), activation='relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    embeddings = layers.Dense(embedding_dim, activation=None)(x)
    model = keras.Model(inputs, embeddings)
    return model

@tf.function
def compute_sphere_loss(embeddings, center):
    return tf.reduce_mean(tf.reduce_sum((embeddings - center)**2, axis=1))

def get_embeddings(model, X_data, batch_size=64):
    emb_list = []
    for i in range(0, len(X_data), batch_size):
        emb = model(X_data[i:i+batch_size], training=False)
        emb_list.append(emb.numpy())
    return np.concatenate(emb_list, axis=0)

def compute_distance(embeddings, center):
    diff = embeddings - center
    dist = np.sqrt(np.sum(diff**2, axis=1))
    return dist

def find_threshold(distances, percentile=95.0):
    return np.percentile(distances, percentile)

def initialize_center(model, X_data, batch_size=64):
    embs = get_embeddings(model, X_data, batch_size)
    c = np.mean(embs, axis=0)
    return tf.Variable(c, dtype=tf.float32)

def train_one_class_model(model, X_data, center, epochs=10, batch_size=64, lr=1e-4):
    optimizer = keras.optimizers.Adam(lr)
    dataset = tf.data.Dataset.from_tensor_slices(X_data)
    dataset = dataset.shuffle(len(X_data)).batch(batch_size)

    for epoch in range(epochs):
        ep_loss = 0.0
        steps = 0
        for batch in dataset:
            with tf.GradientTape() as tape:
                emb = model(batch, training=True)
                loss = compute_sphere_loss(emb, center)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            ep_loss += loss.numpy()
            steps += 1
        print(f"Epoch {epoch+1}/{epochs} - Loss: {ep_loss/steps:.8f}")


# ---------------------------
# (메인) K-Fold (Fold n = train, 나머지 = test) + 모델/Threshold 저장
# ---------------------------
def main_one_class_cv(args):
    date_folders = sorted([d for d in os.listdir(args.base_path) 
                           if os.path.isdir(os.path.join(args.base_path, d))])
    if len(date_folders) < 2:
        raise ValueError("Cross validation을 위해 적어도 2개 이상의 date folder가 필요합니다.")
    print("All date folders:", date_folders)

    fold_accuracies = []
    fold_results = []

    for fold_idx, train_folder in enumerate(date_folders):
        print("="*50)
        print(f"[Fold {fold_idx+1}] Train Folder = {train_folder}")

        test_folders = [f for f in date_folders if f != train_folder]
        print(f" Test Folders = {test_folders}")

        # (1) 학습데이터 (train_folder의 정상='0'만)
        train_structure = get_temporary_structure_with_stable_segments(
            args.base_path, [train_folder], [args.normal_label],
            args.stability_threshold, args.buffer_size
        )
        try:
            X_train = process_images(train_structure, args.img_roi, args.equalize)
        except ValueError:
            print("No training data in this fold. Skipping...")
            fold_accuracies.append(np.nan)
            continue
        print("X_train shape:", X_train.shape)

        # (2) 모델 생성 + Center 초기화 + 학습
        model = create_one_class_model((args.img_roi, args.img_roi, 1), embedding_dim=args.embedding_dim)
        center = initialize_center(model, X_train, batch_size=args.batch_size)
        train_one_class_model(model, X_train, center,
                              epochs=args.epochs, batch_size=args.batch_size, lr=args.learning_rate)

        # (3) threshold 설정
        emb_train = get_embeddings(model, X_train)
        dist_train = compute_distance(emb_train, center)
        threshold = find_threshold(dist_train, percentile=args.threshold_percentile)
        print(f"Fold {fold_idx+1} - Chosen threshold = {threshold:.4f}")

        # -- 모델/center/threshold Fold별 저장 --
        fold_model_path = os.path.join(args.models_dir, f"Fold_{fold_idx+1}_model.h5")
        fold_center_path = os.path.join(args.models_dir, f"Fold_{fold_idx+1}_center.npy")
        fold_thresh_path = os.path.join(args.models_dir, f"Fold_{fold_idx+1}_threshold.txt")

        model.save(fold_model_path)
        np.save(fold_center_path, center.numpy())
        with open(fold_thresh_path, 'w') as f:
            f.write(str(threshold))

        # (필요시) TFLite 변환도 fold별로 저장 가능
        # converter = tf.lite.TFLiteConverter.from_keras_model(model)
        # tflite_model = converter.convert()
        # with open(os.path.join(args.models_dir, f"Fold_{fold_idx+1}_model.tflite"), 'wb') as ft:
        #     ft.write(tflite_model)

        # (4) 테스트
        test_structure = get_temporary_structure_with_stable_segments(
            args.base_path, test_folders, [args.normal_label, args.anomaly_label],
            args.stability_threshold, args.buffer_size
        )
        # 정상 테스트
        test_struc_norm = {
            d_f: {args.normal_label: videos}
            for d_f, subf in test_structure.items()
            for s, videos in subf.items() if s == args.normal_label
        }
        try:
            X_test_normal = process_images(test_struc_norm, args.img_roi, args.equalize)
            y_test_normal = np.zeros(len(X_test_normal), dtype=int)
        except ValueError:
            X_test_normal = np.array([])
            y_test_normal = np.array([])

        # 이상치 테스트
        test_struc_anom = {
            d_f: {args.anomaly_label: videos}
            for d_f, subf in test_structure.items()
            for s, videos in subf.items() if s == args.anomaly_label
        }
        try:
            X_test_anom = process_images(test_struc_anom, args.img_roi, args.equalize)
            y_test_anom = np.ones(len(X_test_anom), dtype=int)
        except ValueError:
            X_test_anom = np.array([])
            y_test_anom = np.array([])

        X_test_all = np.concatenate([X_test_normal, X_test_anom], axis=0) if X_test_anom.size else X_test_normal
        y_test_all = np.concatenate([y_test_normal, y_test_anom], axis=0) if X_test_anom.size else y_test_normal

        if len(X_test_all) == 0:
            print("No test data in this fold. Skipping evaluation.")
            fold_accuracies.append(np.nan)
            continue

        emb_test = get_embeddings(model, X_test_all)
        dist_test = compute_distance(emb_test, center)
        y_pred = (dist_test > threshold).astype(int)  # 1=이상치

        cm = confusion_matrix(y_test_all, y_pred)
        print("Confusion Matrix:\n", cm)
        report = classification_report(y_test_all, y_pred, digits=4, output_dict=True)
        acc = report['accuracy']
        fold_accuracies.append(acc)

        # 결과 로그
        print(f"Fold {fold_idx+1} - Accuracy: {acc:.4f}")
        fold_results.append({
            'fold': fold_idx+1,
            'train_folder': train_folder,
            'test_folders': test_folders,
            'accuracy': acc,
            'confusion_matrix': cm.tolist()
        })

        del model
        gc.collect()
        tf.keras.backend.clear_session()

    # 전체 Fold 요약
    mean_acc = np.nanmean(fold_accuracies)
    std_acc = np.nanstd(fold_accuracies)
    print("="*50)
    print("Cross Validation Results:")
    for r in fold_results:
        print(f"  Fold={r['fold']}, Train={r['train_folder']}, "
              f"Test={r['test_folders']}, Acc={r['accuracy']:.4f}")
    print(f"Mean Accuracy = {mean_acc:.4f}, Std = {std_acc:.4f}")

    if args.save_excel:
        df = pd.DataFrame(fold_results)
        df.to_excel(os.path.join(args.models_dir, "one_class_cv_results.xlsx"), index=False)
        print(f"Results saved to: {os.path.join(args.models_dir, 'one_class_cv_results.xlsx')}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="One-Class CV w/ model saving per fold.")
    parser.add_argument('--base_path', type=str,
        default='/home/jijang/ssd_data/projects/Bacteria/dataset/stable',
        help='Base dataset directory (contains date_folders).')
    parser.add_argument('--models_dir', type=str,
        default='/home/jijang/ssd_data/projects/Bacteria/models/stable_svdd',
        help='Folder to save fold-wise models/centers/thresholds.')
    parser.add_argument('--normal_label', type=str, default='0',
        help='Subfolder name for normal data.')
    parser.add_argument('--anomaly_label', type=str, default='1',
        help='Subfolder name for anomaly data.')
    parser.add_argument('--img_roi', type=int, default=96,
        help='ROI size for images.')
    parser.add_argument('--stability_threshold', type=int, default=350,
        help='Threshold for stable segment detection.')
    parser.add_argument('--buffer_size', type=int, default=2,
        help='Buffer around peaks for stable segments.')
    parser.add_argument('--embedding_dim', type=int, default=32,
        help='Dim of final embedding.')
    parser.add_argument('--epochs', type=int, default=100,
        help='Epochs to train one-class model.')
    parser.add_argument('--batch_size', type=int, default=64,
        help='Batch size.')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
        help='Learning rate.')
    parser.add_argument('--threshold_percentile', type=float, default=95.0,
        help='Set threshold from train distance distribution.')
    parser.add_argument('--equalize', action='store_true',
        help='Apply histogram equalization to images.')
    parser.add_argument('--save_excel', action='store_true',
        help='Save final results to Excel.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    os.makedirs(args.models_dir, exist_ok=True)
    main_one_class_cv(args)
