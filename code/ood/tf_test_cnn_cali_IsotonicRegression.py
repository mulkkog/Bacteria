import os
# GPU 설정
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras import layers, Model
from sklearn.metrics import confusion_matrix
from collections import Counter
from scipy.signal import find_peaks
import pandas as pd
from tensorflow.keras import backend as K
from PIL import Image
from skimage import exposure
from tqdm import tqdm

# 추가: IsotonicRegression 임포트
from sklearn.isotonic import IsotonicRegression

##############################################################################
# 1) Isotonic Regression Calibration Functions (Temperature Scaling 대신)
##############################################################################

def fit_isotonic_regression(uncalibrated_probs, y_true, num_classes):
    """
    Fit isotonic regression models for each class (one-vs-rest) using the predicted probabilities and true labels.
    uncalibrated_probs: (N, num_classes) predicted probabilities from the model.
    y_true: (N,) or (N, num_classes) one-hot encoded true labels.
    Returns a list of fitted IsotonicRegression models, one for each class.
    """
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    iso_models = []
    for i in range(num_classes):
        prob_i = uncalibrated_probs[:, i]
        y_bin = (y_true == i).astype(np.float32)
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(prob_i, y_bin)
        iso_models.append(ir)
    return iso_models

def apply_isotonic_regression(uncalibrated_probs, iso_models, num_classes):
    """
    Apply the fitted isotonic regression models to calibrate the predicted probabilities.
    Returns calibrated probabilities normalized to sum to 1 for each sample.
    """
    calibrated_probs = np.zeros_like(uncalibrated_probs)
    for i in range(num_classes):
        calibrated_probs[:, i] = iso_models[i].predict(uncalibrated_probs[:, i])
    # Renormalize each row to sum to 1
    calibrated_probs = calibrated_probs / np.sum(calibrated_probs, axis=1, keepdims=True)
    return calibrated_probs


def compute_ece(probabilities, labels, n_bins=10):
    """
    Expected Calibration Error (ECE) 계산 함수.
    probabilities: (N, num_classes) 예측 확률
    labels: (N,) 정답 (int)
    n_bins: calibration bin 개수
    """
    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0
    N = len(labels)

    for i in range(n_bins):
        start = bin_boundaries[i]
        end = bin_boundaries[i+1]

        bin_mask = (confidences >= start) & (confidences < end)
        bin_size = np.sum(bin_mask)

        if bin_size > 0:
            avg_conf = np.mean(confidences[bin_mask])
            accuracy_bin = np.mean(predictions[bin_mask] == labels[bin_mask])
            prop_bin = bin_size / N
            ece += np.abs(avg_conf - accuracy_bin) * prop_bin

    return ece

##############################################################################
# 2) 이미 학습된 Sequential 모델의 "마지막 softmax 제거" 함수
##############################################################################

def remove_final_softmax_sequential(model):
    config = model.get_config()
    if config['layers']:
        last_layer = config['layers'][-1]
        if last_layer['class_name'] == 'Dense':
            if 'softmax' in last_layer['config']['activation']:
                last_layer['config']['activation'] = 'linear'
    new_model = Sequential.from_config(config)
    new_model.set_weights(model.get_weights())
    return new_model

##############################################################################
# 3) 기타 유틸 및 데이터 처리
##############################################################################

def create_calibrated_model(model_no_softmax, T_value):
    """
    기존 Temperature Scaling 기반 보정 모델 생성 함수 - 사용하지 않음.
    """
    inputs = model_no_softmax.input
    logits = model_no_softmax.output
    scaled_logits = layers.Lambda(lambda x: x / T_value, name="temperature_scaling")(logits)
    outputs = layers.Softmax(name="calibrated_softmax")(scaled_logits)
    calibrated_model = Model(inputs=inputs, outputs=outputs, name="calibrated_model")
    return calibrated_model

def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(42)

def load_tiff_images(folder_path, file_names, apply_equalization):
    images = []
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        with Image.open(file_path) as img:
            image = np.array(img).astype(np.float32) / 255.0
            if apply_equalization:
                image = exposure.equalize_hist(image)
            images.append(image)
    return np.array(images)

def load_yuv_images(folder_path, file_names, apply_equalization, image_size=(128, 128)):
    images = []
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'rb') as file:
            image = np.frombuffer(file.read(128 * 128), dtype=np.uint8).reshape(image_size)
            image = image.astype(np.float32) / 255.0
            if apply_equalization:
                image = exposure.equalize_hist(image)
            images.append(image)
    return np.array(images)

def get_images(file_paths, apply_equalization):
    images = []
    for file_path in file_paths:
        if file_path.endswith('.tiff'):
            with Image.open(file_path) as img:
                image = np.array(img).astype(np.float32) / 255.0
                if apply_equalization:
                    image = exposure.equalize_hist(image)
        elif file_path.endswith('.yuv'):
            with open(file_path, 'rb') as image_file:
                image = np.frombuffer(image_file.read(128 * 128), dtype=np.uint8).reshape((128, 128)).astype(np.float32) / 255.0
                if apply_equalization:
                    image = exposure.equalize_hist(image)
        else:
            raise ValueError("Unsupported file format")
        images.append(image)
    return np.array(images)

def preprocess_data(file_paths, img_roi, apply_equalization):
    X = get_images(file_paths, apply_equalization)
    X_tests = abs(X - np.roll(X, -1, axis=0))
    X_tests = np.delete(X_tests, -1, axis=0)
    X_tests = np.expand_dims(X_tests, axis=-1).astype(np.float32)
    X_tests = X_tests[:, :img_roi, :img_roi, :]
    return X_tests

def find_peaks_indices(summed_diffs, height):
    peaks, _ = find_peaks(summed_diffs, height=height)
    return peaks

def find_all_stable_segments(summed_diffs, peaks, buffer_size):
    total_frames = len(summed_diffs)
    excluded_indices = set()
    for peak in peaks:
        for i in range(max(0, peak - buffer_size), min(total_frames, peak + buffer_size + 1)):
            excluded_indices.add(i)
    valid_frames = [i for i in range(total_frames) if i not in excluded_indices]
    return valid_frames

def process_video_files(folder_path, file_names, stability_threshold, buffer_size, apply_equalization=False):
    if not file_names:
        print("The folder is empty. No files to process.")
        return []
    if file_names[0].endswith('.yuv'):
        images = load_yuv_images(folder_path, file_names, apply_equalization)
    elif file_names[0].endswith('.tiff'):
        images = load_tiff_images(folder_path, file_names, apply_equalization)
    else:
        raise ValueError("Unsupported file format")
    image_diffs = np.abs(images[:-1] - images[1:])
    summed_diffs = image_diffs.sum(axis=(1, 2))
    peaks = find_peaks_indices(summed_diffs, height=stability_threshold)
    stable_segments = find_all_stable_segments(summed_diffs, peaks, buffer_size)
    return stable_segments

def get_temporary_structure_with_stable_segments(base_path, date_folders, subfolders, stability_threshold, buffer_size):
    structure = {}
    no_stable_segments_videos = []
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
                            stable_segments = process_video_files(
                                video_folder_path,
                                video_files_sorted,
                                stability_threshold,
                                buffer_size
                            )
                            if stable_segments:
                                structure[date_folder][subfolder][video_name_folder] = {}
                                start = stable_segments[0]
                                segment_id = 0
                                for i in range(1, len(stable_segments)):
                                    if stable_segments[i] != stable_segments[i - 1] + 1:
                                        end = stable_segments[i - 1] + 1
                                        segment_files = [
                                            os.path.join(video_folder_path, video_files_sorted[j])
                                            for j in range(start, end)
                                        ]
                                        structure[date_folder][subfolder][video_name_folder][f"segment_{segment_id}"] = segment_files
                                        segment_id += 1
                                        start = stable_segments[i]
                                end = stable_segments[-1] + 1
                                segment_files = [
                                    os.path.join(video_folder_path, video_files_sorted[j])
                                    for j in range(start, end)
                                ]
                                structure[date_folder][subfolder][video_name_folder][f"segment_{segment_id}"] = segment_files
                            else:
                                no_stable_segments_videos.append(
                                    (date_folder, subfolder, video_name_folder, "No stable segments")
                                )
                                print(f"No stable segments found in {video_name_folder}")
    return structure, no_stable_segments_videos

##############################################################################
# 4) 평가 함수 (Frame 레벨) - ECE 추가 (Temperature Scaling 이전)
##############################################################################

def evaluate_model(model, X_test, y_test, num_classes):
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_test_indices = np.argmax(y_test, axis=1)
    conf_matrix = confusion_matrix(y_test_indices, y_pred, labels=range(num_classes))
    accuracy = np.mean(y_pred == y_test_indices)
    weights = np.array([10 ** i for i in range(num_classes)], dtype=np.float32)
    weights[0] = 0
    row_weighted_scores, row_total_weighted_scores = [], []
    for row in conf_matrix:
        total_samples = np.sum(row)
        if total_samples == 0:
            row_weighted_scores.append([0] * len(weights))
            row_total_weighted_scores.append(0)
            continue
        top1_index = np.argmax(row)
        top1_value = row[top1_index]
        row_excluding_top1 = np.copy(row)
        row_excluding_top1[top1_index] = 0
        top2_index = np.argmax(row_excluding_top1)
        top2_value = row_excluding_top1[top2_index]
        top1_scores = top1_value * weights[top1_index] / total_samples
        if abs(top2_index - top1_index) == 1:
            top2_scores = top2_value * weights[top2_index] / total_samples
            total_weighted_score = top1_scores + top2_scores
        else:
            total_weighted_score = top1_scores
        row_weighted_scores.append([row[i] * weights[i] / total_samples for i in range(len(row))])
        row_total_weighted_scores.append(total_weighted_score)
    ece_before_iso = compute_ece(y_pred_probs, y_test_indices, n_bins=10)
    return accuracy, conf_matrix, row_weighted_scores, row_total_weighted_scores, ece_before_iso

##############################################################################
# 5) Video 레벨 평가 (TFLite 모델로, TS 미적용) - 기존 코드 그대로
##############################################################################

def model_inference(X_test, tflite_model_path):
    interpreter = tf.lite.Interpreter(tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    output_data = []
    for i in range(X_test.shape[0]):
        single_input = np.expand_dims(X_test[i], axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], single_input)
        interpreter.invoke()
        single_output = interpreter.get_tensor(output_details[0]['index'])
        single_output = np.argmax(single_output, axis=1)
        output_data.append(single_output)
    return np.array(output_data)

def find_mode(lst):
    lst = lst.reshape(-1)
    counter = Counter(lst)
    return counter.most_common(1)[0][0]

def evaluate_videos(structure, tflite_model_path, img_roi, num_classes, apply_equalization):
    confusion_matrix_result = np.zeros((num_classes, num_classes), dtype=int)
    video_results = {}
    actual_classes = {}
    insufficient_frame_videos = set()
    for date_folder, subfolders in structure.items():
        for subfolder, video_folders in subfolders.items():
            for video_folder, segments in video_folders.items():
                insufficient_frames_count = 0
                for segment_id, file_paths in segments.items():
                    if (file_paths is None) or (len(file_paths) <= 1):
                        insufficient_frames_count += 1
                        continue
                    X_test_seg = preprocess_data(file_paths, img_roi, apply_equalization)
                    result = model_inference(X_test_seg, tflite_model_path)
                    most_common_result = find_mode(result)
                    if video_folder not in video_results:
                        video_results[video_folder] = []
                        actual_classes[video_folder] = int(subfolder)
                    video_results[video_folder].append(most_common_result)
                if insufficient_frames_count == len(segments):
                    insufficient_frame_videos.add((date_folder, subfolder, video_folder, "Insufficient frames"))
    for video_folder, results in video_results.items():
        final_class = find_mode(np.array(results))
        subfolder_int = actual_classes[video_folder]
        confusion_matrix_result[subfolder_int, final_class] += 1
    return confusion_matrix_result, list(insufficient_frame_videos)

##############################################################################
# 6) Video 레벨에서 Isotonic Regression 적용
##############################################################################

def evaluate_videos_with_isotonic_regression_batch_ece(structure, model, iso_models, img_roi, num_classes, apply_equalization):
    """
    Video 레벨 평가 (Isotonic Regression 적용) + Video 레벨 ECE 계산.
    1) 모든 세그먼트 프레임을 모아 X_all에 쌓은 뒤 한 번에 predict (배치 추론)
    2) 모델 예측 확률 -> Isotonic Regression 적용 -> (N, num_classes)
    3) 프레임별 확률을 세그먼트/비디오별로 재그룹핑
    4) Video 확률 벡터 = 모든 프레임 확률의 평균
    5) Video별 argmax로 Confusion Matrix 계산
    6) Video별 confidence로 ECE 계산
    """
    from collections import defaultdict
    X_all = []
    segment_info_list = []
    for date_folder, subfolders in structure.items():
        for subfolder, video_folders in subfolders.items():
            for video_folder, segments in video_folders.items():
                for segment_id, file_paths in segments.items():
                    if (file_paths is None) or (len(file_paths) <= 1):
                        continue
                    X_test_seg = preprocess_data(file_paths, img_roi, apply_equalization)
                    if X_test_seg.shape[0] == 0:
                        continue
                    start_idx = len(X_all)
                    X_all.extend(X_test_seg)
                    end_idx = len(X_all)
                    segment_info_list.append((date_folder, subfolder, video_folder, segment_id, start_idx, end_idx))
    if len(X_all) == 0:
        confusion_matrix_result = np.zeros((num_classes, num_classes), dtype=int)
        return confusion_matrix_result, 0.0, []
    X_all = np.array(X_all, dtype=np.float32)
    uncalibrated_probs = model.predict(X_all, batch_size=64)
    calibrated_probs_all = apply_isotonic_regression(uncalibrated_probs, iso_models, num_classes)
    from collections import defaultdict
    video_prob_sums = defaultdict(lambda: np.zeros(num_classes, dtype=np.float32))
    video_frame_counts = defaultdict(int)
    for (date_folder, subfolder, video_folder, segment_id, start_idx, end_idx) in segment_info_list:
        key = (date_folder, subfolder, video_folder)
        video_prob_sums[key] += np.sum(calibrated_probs_all[start_idx:end_idx], axis=0)
        video_frame_counts[key] += (end_idx - start_idx)
    confusion_matrix_result = np.zeros((num_classes, num_classes), dtype=int)
    video_confidences = []
    video_labels = []
    for (date_folder, subfolder, video_folder), prob_sum in video_prob_sums.items():
        count = video_frame_counts[(date_folder, subfolder, video_folder)]
        if count == 0:
            continue
        prob_video = prob_sum / float(count)
        pred_class = np.argmax(prob_video)
        true_class = int(subfolder)
        confusion_matrix_result[true_class, pred_class] += 1
        confidence = np.max(prob_video)
        video_confidences.append(confidence)
        video_labels.append(true_class == pred_class)
    num_videos = len(video_confidences)
    prob_for_ece = np.zeros((num_videos, num_classes), dtype=np.float32)
    video_true_classes = []
    idx = 0
    for (date_folder, subfolder, video_folder), prob_sum in video_prob_sums.items():
        count = video_frame_counts[(date_folder, subfolder, video_folder)]
        if count == 0:
            continue
        prob_video = prob_sum / float(count)
        pred_class = np.argmax(prob_video)
        confidence = np.max(prob_video)
        video_true_classes.append(int(subfolder))
        prob_for_ece[idx, pred_class] = confidence
        idx += 1
    video_true_classes = np.array(video_true_classes, dtype=int)
    video_ece = compute_ece(prob_for_ece, video_true_classes, n_bins=10)
    return confusion_matrix_result, video_ece, []

##############################################################################
# 7) Frame 레벨 Isotonic Regression Calibration
##############################################################################

def evaluate_model_with_isotonic_regression(model, X_test, y_test, num_classes):
    """
    Frame 레벨 평가 (Isotonic Regression 적용)
    1) 모델의 예측 확률 추출
    2) Isotonic Regression 모델 학습 (one-vs-rest 방식)
    3) 보정된 확률 계산 및 ECE, Confusion Matrix 평가
    """
    uncalibrated_probs = model.predict(X_test)
    iso_models = fit_isotonic_regression(uncalibrated_probs, y_test, num_classes)
    calibrated_probs = apply_isotonic_regression(uncalibrated_probs, iso_models, num_classes)
    y_test_indices = np.argmax(y_test, axis=1)
    ece_iso = compute_ece(calibrated_probs, y_test_indices, n_bins=10)
    y_pred_iso = np.argmax(calibrated_probs, axis=1)
    conf_matrix_iso = confusion_matrix(y_test_indices, y_pred_iso, labels=range(num_classes))
    accuracy_iso = np.mean(y_pred_iso == y_test_indices)
    weights = np.array([10 ** i for i in range(num_classes)], dtype=np.float32)
    weights[0] = 0
    row_weighted_scores_iso, row_total_weighted_scores_iso = [], []
    for row in conf_matrix_iso:
        total_samples = np.sum(row)
        if total_samples == 0:
            row_weighted_scores_iso.append([0]*len(weights))
            row_total_weighted_scores_iso.append(0)
            continue
        top1_index = np.argmax(row)
        top1_value = row[top1_index]
        row_excluding_top1 = np.copy(row)
        row_excluding_top1[top1_index] = 0
        top2_index = np.argmax(row_excluding_top1)
        top2_value = row_excluding_top1[top2_index]
        top1_scores = top1_value * weights[top1_index] / total_samples
        if abs(top2_index - top1_index) == 1:
            top2_scores = top2_value * weights[top2_index] / total_samples
            total_weighted_score = top1_scores + top2_scores
        else:
            total_weighted_score = top1_scores
        row_weighted_scores_iso.append([row[i] * weights[i] / total_samples for i in range(len(row))])
        row_total_weighted_scores_iso.append(total_weighted_score)
    return (
        accuracy_iso,
        conf_matrix_iso,
        ece_iso,
        row_weighted_scores_iso,
        row_total_weighted_scores_iso,
        iso_models
    )

##############################################################################
# 8) Main 함수
##############################################################################

def calculate_cutoff_accuracy(conf_matrix):
    tp_top_left = np.sum(conf_matrix[0:2, 0:2])
    tp_bottom_right = np.sum(conf_matrix[2:, 2:])
    total_true_positives = tp_top_left + tp_bottom_right
    total_samples = np.sum(conf_matrix)
    accuracy = total_true_positives / total_samples if total_samples else 0
    return accuracy

def calculate_f1_scores(conf_matrix):
    f1_scores = []
    num_classes = conf_matrix.shape[0]
    for i in range(num_classes):
        tp = conf_matrix[i, i]
        fp = conf_matrix[:, i].sum() - tp
        fn = conf_matrix[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    return f1_scores
    
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
                        y_data.append(int(subfolder))
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

def main(args):
    os.makedirs(args.excel_dir, exist_ok=True)
    excel_path = os.path.join(args.excel_dir, "test_results.xlsx")
    pd.DataFrame().to_excel(excel_path, index=False)
    calibrated_models_dir = args.models_dir + "_cali"
    os.makedirs(calibrated_models_dir, exist_ok=True)

    date_folders = sorted([d for d in os.listdir(args.base_path) if os.path.isdir(os.path.join(args.base_path, d))])
    print(f"Total date folders found: {date_folders}")

    if not date_folders:
        raise ValueError("No date folders found in the base path. Please check the dataset directory.")

    results = pd.DataFrame(columns=[
        'Fold', 'Train Folders', 'Test Folder',
        'Frame Confusion Matrix', 'Frame Accuracy',
        'ECE before Iso',                 # 수정됨
        'Video Confusion Matrix', 'Video Accuracy',
        'Cut-off 10^2 CFU/ml', 'Frame F1 Score', 'Video F1 Score',
        'Row Weighted Scores', 'Row Total Weighted Scores',
        'Iso Frame Accuracy', 'Iso Frame Confusion Matrix', 'ECE after Iso',
        'Iso Video Accuracy', 'Iso Video Confusion Matrix'
    ])

    cumulative_frame_conf_matrix = np.zeros((args.num_classes, args.num_classes))
    cumulative_video_conf_matrix = np.zeros((args.num_classes, args.num_classes))
    cumulative_iso_video_conf_matrix = np.zeros((args.num_classes, args.num_classes))

    no_stable_segments_videos_all_folds = []

    for fold_index, test_folder in enumerate(date_folders):
        print(f"Starting Fold {fold_index + 1} - Test Folder: {test_folder}")
        val_folder = [test_folder]
        train_folders = [folder for folder in date_folders if folder != test_folder]

        print(f"Train Folders: {train_folders}")
        print(f"Validation Folder: {val_folder}")

        train_structure, _ = get_temporary_structure_with_stable_segments(
            args.base_path, train_folders, args.subfolders, args.stability_threshold, args.buffer_size
        )
        val_structure, no_stable_segments_videos = get_temporary_structure_with_stable_segments(
            args.base_path, val_folder, args.subfolders, args.stability_threshold, args.buffer_size
        )
        no_stable_segments_videos_all_folds.append((fold_index + 1, no_stable_segments_videos))

        try:
            X_test, y_test = process_images(val_structure, args.img_roi, args.num_classes, args.equalize)
        except ValueError as e:
            print(f"Error processing images for Fold {fold_index + 1}: {e}")
            continue

        if len(X_test) == 0:
            print(f"No data available for Fold {fold_index + 1}. Skipping.")
            continue

        print(f"Fold {fold_index + 1}: Test data shape: {X_test.shape}, Test labels shape: {y_test.shape}")

        model_path = os.path.join(args.models_dir, f"Fold_{fold_index + 1}_model.h5")
        if not os.path.exists(model_path):
            print(f"Model file {model_path} does not exist. Skipping Fold {fold_index + 1}.")
            continue
        model = load_model(model_path)

        trainable_params = np.sum([K.count_params(w) for w in model.trainable_weights])
        print(f"Fold {fold_index + 1}: Number of trainable parameters: {trainable_params}")
        model_size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"Fold {fold_index + 1}: Model size: {model_size:.2f} MB")

        (frame_accuracy,
         frame_conf_matrix,
         row_weighted_scores,
         row_total_weighted_scores,
         frame_ece_before_iso
        ) = evaluate_model(model, X_test, y_test, args.num_classes)

        cumulative_frame_conf_matrix += frame_conf_matrix

        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS
            ]
            tflite_model = converter.convert()
            tflite_model_path = os.path.join(args.models_dir, f"Fold_{fold_index + 1}_HTCNN.tflite")
            with open(tflite_model_path, 'wb') as f:
                f.write(tflite_model)
            print(f"TFLite 모델이 성공적으로 저장되었습니다: {tflite_model_path}")
        except Exception as e:
            print(f"TFLite 변환 중 오류가 발생했습니다: {e}")
            tflite_model_path = None

        video_conf_matrix = np.zeros((args.num_classes, args.num_classes))
        if tflite_model_path is not None:
            video_conf_matrix, additional_no_stable_segments_videos = evaluate_videos(
                val_structure, tflite_model_path, args.img_roi, args.num_classes, args.equalize
            )
            no_stable_segments_videos_all_folds[-1][1].extend(additional_no_stable_segments_videos)
            cumulative_video_conf_matrix += video_conf_matrix
        
        video_accuracy = np.trace(video_conf_matrix) / np.sum(video_conf_matrix) if np.sum(video_conf_matrix) else 0

        print(f"Evaluating Fold {fold_index + 1}: test {test_folder}")
        print(f"Frame Confusion Matrix:\n{frame_conf_matrix}")
        print(f"Frame Accuracy: {frame_accuracy * 100:.2f}%")
        print(f"Frame ECE (before Iso): {frame_ece_before_iso:.4f}\n")

        print(f"Video Confusion Matrix:\n{video_conf_matrix}")
        print(f"Video Accuracy: {video_accuracy * 100:.2f}%\n")

        frame_f1_scores = calculate_f1_scores(frame_conf_matrix)
        video_f1_scores = calculate_f1_scores(video_conf_matrix)

        if len(args.subfolders) == 5:
            accuracy_cutoff_revised = calculate_cutoff_accuracy(video_conf_matrix)
        else:
            accuracy_cutoff_revised = None

        (iso_frame_accuracy,
         iso_frame_conf_matrix,
         ece_iso,
         row_weighted_scores_iso,
         row_total_weighted_scores_iso,
         iso_models
        ) = evaluate_model_with_isotonic_regression(model, X_test, y_test, args.num_classes)

        print(f"[Isotonic Regression] Fold {fold_index+1}")
        print(f"[Iso] Frame Accuracy: {iso_frame_accuracy*100:.2f}%")
        print(f"[Iso] ECE: {ece_iso:.4f}")
        print(f"[Iso] Confusion Matrix:\n{iso_frame_conf_matrix}\n")

        iso_video_conf_matrix, iso_video_ece, _ = evaluate_videos_with_isotonic_regression_batch_ece(
            val_structure, 
            model,
            iso_models,
            args.img_roi,
            args.num_classes,
            args.equalize
        )
        # 참고: Isotonic Regression 보정 모델은 Keras 모델로 저장하기 어렵습니다.
        iso_video_accuracy = np.trace(iso_video_conf_matrix) / np.sum(iso_video_conf_matrix) if np.sum(iso_video_conf_matrix) else 0
        cumulative_iso_video_conf_matrix += iso_video_conf_matrix

        print("Iso Video Accuracy:", iso_video_accuracy)
        print("Iso Video ECE:", iso_video_ece)
        print("Iso Video Confusion Matrix:\n", iso_video_conf_matrix)

        fold_data = {
            'Fold': fold_index + 1,
            'Train Folders': ', '.join(train_folders),
            'Test Folder': test_folder if isinstance(test_folder, str) else str(test_folder),
            'Frame Confusion Matrix': frame_conf_matrix.tolist(),
            'Frame Accuracy': frame_accuracy,
            'ECE before Iso': frame_ece_before_iso,
            'Video Confusion Matrix': video_conf_matrix.tolist(),
            'Video Accuracy': video_accuracy,
            'Cut-off 10^2 CFU/ml': accuracy_cutoff_revised if accuracy_cutoff_revised is not None else "N/A",
            'Frame F1 Score': frame_f1_scores,
            'Video F1 Score': video_f1_scores,
            'Row Weighted Scores': [list(ws) for ws in row_weighted_scores],
            'Row Total Weighted Scores': row_total_weighted_scores,
            'Iso Frame Accuracy': iso_frame_accuracy,
            'Iso Frame Confusion Matrix': iso_frame_conf_matrix.tolist(),
            'ECE after Iso': ece_iso,
            'Iso Video Accuracy': iso_video_accuracy,
            'Iso Video Confusion Matrix': iso_video_conf_matrix.tolist()
        }

        results = pd.concat([results, pd.DataFrame([fold_data])], ignore_index=True)

    total_frame_accuracy = (np.trace(cumulative_frame_conf_matrix) / np.sum(cumulative_frame_conf_matrix)
                            if np.sum(cumulative_frame_conf_matrix) else 0)
    total_video_accuracy = (np.trace(cumulative_video_conf_matrix) / np.sum(cumulative_video_conf_matrix)
                            if np.sum(cumulative_video_conf_matrix) else 0)
    total_iso_video_accuracy = (np.trace(cumulative_iso_video_conf_matrix) / np.sum(cumulative_iso_video_conf_matrix)
                               if np.sum(cumulative_iso_video_conf_matrix) else 0)

    total_frame_f1_score = calculate_f1_scores(cumulative_frame_conf_matrix)
    total_video_f1_score = calculate_f1_scores(cumulative_video_conf_matrix)
    total_iso_video_f1_score = calculate_f1_scores(cumulative_iso_video_conf_matrix)

    print("Total Frame Confusion Matrix:\n", cumulative_frame_conf_matrix.astype(int))
    print(f"Total Frame Accuracy: {total_frame_accuracy * 100:.2f}%")
    print(f"Total Frame F1 Score: {total_frame_f1_score}\n")

    print("Total Video Confusion Matrix:\n", cumulative_video_conf_matrix.astype(int))
    print(f"Total Video Accuracy: {total_video_accuracy * 100:.2f}%")
    print(f"Total Video F1 Score: {total_video_f1_score}\n")

    print("Total Iso Video Confusion Matrix:\n", cumulative_iso_video_conf_matrix.astype(int))
    print(f"Total Iso Video Accuracy: {total_iso_video_accuracy * 100:.2f}%")
    print(f"Total Iso Video F1 Score: {total_iso_video_f1_score}\n")

    results.to_excel(excel_path, index=False)
    with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a') as writer:
        df_summary = pd.DataFrame({
            'Metric': [
                'Total Frame Accuracy',
                'Total Video Accuracy',
                'Total Iso Video Accuracy'
            ],
            'Value': [
                f"{total_frame_accuracy*100:.2f}%",
                f"{total_video_accuracy*100:.2f}%",
                f"{total_iso_video_accuracy*100:.2f}%"
            ]
        })
        df_summary.to_excel(writer, sheet_name='Overall Summary', index=False)

    print(f"\nResults saved to: {excel_path}")
    print("\nNo-stable-segments videos by fold:")
    for fold_idx, videos in no_stable_segments_videos_all_folds:
        if videos:
            print(f" Fold {fold_idx}:")
            for v in videos:
                date_folder, subfolder, video_name, reason = v
                print(f"   - {date_folder}/{subfolder}/{video_name}: {reason}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Script to evaluate a model with Isotonic Regression Calibration (both Frame and Video levels).")
    parser.add_argument('--base_path', type=str, default='/home/jijang/ssd_data/projects/Bacteria/dataset/stable', help='Base dataset directory.')
    parser.add_argument('--models_dir', type=str, default='/home/jijang/ssd_data/projects/Bacteria/models/2025_bacteria_stable', help='Directory where .h5 models are saved.')
    parser.add_argument('--excel_dir', type=str, default='/home/jijang/ssd_data/projects/Bacteria/excel/2025_bacteria_stable', help='Directory for saving results as excel.')
    parser.add_argument('--subfolders', type=str, nargs='+', default=['0', '1'], help='Subfolders as classes.')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes.')
    parser.add_argument('--img_frame', type=int, default=300)
    parser.add_argument('--img_roi', type=int, default=96)
    parser.add_argument('--stability_threshold', type=int, default=350)
    parser.add_argument('--buffer_size', type=int, default=2)
    parser.add_argument('--equalize', action='store_true', help='Apply histogram equalization.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
