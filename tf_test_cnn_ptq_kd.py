import os
# GPU 설정
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "5"

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from collections import Counter
from scipy.signal import find_peaks
import pandas as pd
from tensorflow.keras import backend as K
from PIL import Image
from skimage import exposure
from tqdm import tqdm
from tensorflow.keras import losses 

# --- 추가: convert_variables_to_constants_v2 임포트 (MACs, FLOPs 계산에 사용) ---
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(42)

# ----------------------------------------------------------------------------
# 공통: 모델 파일 크기를 계산하는 함수
def get_model_size(file_path):
    """
    주어진 파일 경로에 있는 모델 파일의 크기를 계산하여,
    바이트, 킬로바이트, 메가바이트 단위로 반환합니다.
    """
    size_bytes = os.path.getsize(file_path)
    size_kb = size_bytes / 1024
    size_mb = size_kb / 1024
    return size_bytes, size_kb, size_mb

# ----------------------------------------------------------------------------
# 추가: .h5 모델에 대해 MACs, FLOPs, 파라미터 수, 모델 크기를 계산하는 함수
def get_h5_model_metrics(model, input_shape, model_file_path):
    """
    model: 로드된 keras 모델 (.h5)
    input_shape: 단일 입력 shape, 예) (96, 96, 1)
    model_file_path: 모델이 저장된 파일 경로 (.h5)
    """
    # 모델을 함수 형태로 변환하여 concrete function 생성
    concrete_func = tf.function(lambda x: model(x))
    input_spec = tf.TensorSpec([1] + list(input_shape), model.inputs[0].dtype)
    concrete_func = concrete_func.get_concrete_function(input_spec)
    
    # 모델을 상수 그래프로 변환 (이제 반환값은 하나의 객체)
    frozen_func = convert_variables_to_constants_v2(concrete_func)
    graph_def = frozen_func.graph.as_graph_def()
    
    # FLOPs 계산 (모든 부동소수점 연산 수)
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops_profile = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
        flops = flops_profile.total_float_ops if flops_profile is not None else 0
    # 일반적으로 1 MAC = 2 FLOPs
    macs = flops / 2

    # 모델 파라미터 수
    num_params = model.count_params()
    # 모델 파일 크기 (바이트)
    model_size = os.path.getsize(model_file_path)
    return macs, flops, num_params, model_size


# ----------------------------------------------------------------------------
# 이하 기존 코드 (영상/이미지 전처리, 모델 추론 및 평가 함수)

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
                        y_data.append(subfolder)  # 클래스 라벨로 서브폴더 이름 사용
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
    no_stable_segments_videos = []  # 안정적인 세그먼트가 없는 비디오 저장 리스트

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
                                # 마지막 세그먼트 추가
                                end = stable_segments[-1] + 1
                                segment_files = [os.path.join(video_folder_path, video_files_sorted[j]) for j in range(start, end)]
                                structure[date_folder][subfolder][video_name_folder][f"segment_{segment_id}"] = segment_files
                            else:
                                no_stable_segments_videos.append((date_folder, subfolder, video_name_folder, "No stable segments"))
                                print(f"No stable segments found in {video_name_folder}")
    return structure, no_stable_segments_videos

# TIFF 이미지 로드 함수
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

# YUV 이미지 로드 함수
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

# 비디오 파일 처리 함수
def process_video_files(folder_path, file_names, stability_threshold, buffer_size, apply_equalization=False):
    if not file_names:
        print("The folder is empty. No files to process.")
        return []
        
    # 파일 포맷 결정
    if file_names[0].endswith('.yuv'):
        images = load_yuv_images(folder_path, file_names, apply_equalization)
    elif file_names[0].endswith('.tiff'):
        images = load_tiff_images(folder_path, file_names, apply_equalization)
    else:
        raise ValueError("Unsupported file format")

    # 이미지 간 차이 계산
    image_diffs = np.abs(images[:-1] - images[1:])
    summed_diffs = image_diffs.sum(axis=(1, 2))

    # 피크 찾기
    peaks, _ = find_peaks(summed_diffs, height=stability_threshold)

    # 안정적인 세그먼트 찾기
    stable_segments = find_all_stable_segments(summed_diffs, peaks, buffer_size)
    return stable_segments

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

# TFLite 모델 추론 함수
def model_inference(X_test, tflite_model_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    output_data = []
    for i in range(X_test.shape[0]):
        single_input = np.expand_dims(X_test[i], axis=0)  # 원래 float32 형태
        
        # 만약 모델의 입력 dtype가 int8이라면 양자화 진행
        if input_details[0]['dtype'] == np.int8:
            input_scale, input_zero_point = input_details[0]['quantization']
            single_input = (single_input / input_scale + input_zero_point).astype(np.int8)
        
        interpreter.set_tensor(input_details[0]['index'], single_input)
        interpreter.invoke()
        single_output = interpreter.get_tensor(output_details[0]['index'])
        
        # 만약 출력이 int8이라면, dequantize 진행
        if output_details[0]['dtype'] == np.int8:
            output_scale, output_zero_point = output_details[0]['quantization']
            single_output = (single_output.astype(np.float32) - output_zero_point) * output_scale
        
        single_output = np.argmax(single_output, axis=1)
        output_data.append(single_output)
        
    return np.array(output_data)

def find_mode(lst):
    lst = lst.reshape(-1)
    counter = Counter(lst)
    most_common = counter.most_common(1)
    return most_common[0][0]

def evaluate_model_tflite(tflite_model_path, X_test, y_test, num_classes):
    # TFLite 모델을 이용해 프레임 단위 추론 진행
    y_pred = model_inference(X_test, tflite_model_path)
    y_pred = np.array(y_pred).flatten()  # shape (N,)
    y_true = np.argmax(y_test, axis=1)
    
    conf_matrix = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    
    weights = np.array([10 ** i for i in range(num_classes)], dtype=np.float32)
    weights[0] = 0
    
    accuracy = np.mean(y_pred == y_true)
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

    return accuracy, conf_matrix, row_weighted_scores, row_total_weighted_scores

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

                    images = preprocess_data(file_paths, img_roi, apply_equalization)
                    result = model_inference(images, tflite_model_path)
                    most_common_result = find_mode(result)

                    if video_folder not in video_results:
                        video_results[video_folder] = []
                        actual_classes[video_folder] = subfolder
                    video_results[video_folder].append(most_common_result)

                if insufficient_frames_count == len(segments):
                    insufficient_frame_videos.add((date_folder, subfolder, video_folder, "Insufficient frames"))

    for video_folder, results in video_results.items():
        final_class = find_mode(np.array(results))
        subfolder = actual_classes[video_folder]
        confusion_matrix_result[int(subfolder), final_class] += 1

    return confusion_matrix_result, list(insufficient_frame_videos)

def preprocess_data(file_paths, img_roi, apply_equalization):
    X = get_images(file_paths, apply_equalization)
    X = np.array(X)
    X_tests = abs(X - np.roll(X, -1, axis=0))
    X_tests = np.delete(X_tests, -1, axis=0)
    X_test = np.expand_dims(X_tests, axis=-1)
    X_test = np.asarray(X_test).astype(np.float32)
    X_test = X_test[:, :img_roi, :img_roi, :]
    return X_test

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
 
class DistillationLoss(losses.Loss):
    def __init__(self, teacher_predictions=None, temperature=3, alpha=0.1, 
                 reduction=losses.Reduction.AUTO, name='distillation_loss', **kwargs):
        # teacher_predictions 기본값을 None으로 설정합니다.
        super(DistillationLoss, self).__init__(reduction=reduction, name=name, **kwargs)
        self.teacher_predictions = teacher_predictions
        self.temperature = temperature
        self.alpha = alpha

    def call(self, y_true, y_pred):
        # teacher_predictions가 제공되지 않은 경우 hard loss만 반환합니다.
        if self.teacher_predictions is None:
            return losses.categorical_crossentropy(y_true, y_pred)
        else:
            batch_size = tf.shape(y_pred)[0]
            teacher_pred_batch = tf.slice(self.teacher_predictions, [0, 0], [batch_size, -1])
            soft_teacher_pred = tf.nn.softmax(teacher_pred_batch / self.temperature)
            soft_student_pred = tf.nn.softmax(y_pred / self.temperature)
            distillation_loss = losses.KLDivergence()(soft_teacher_pred, soft_student_pred)
            hard_loss = losses.categorical_crossentropy(y_true, y_pred)
            return self.alpha * distillation_loss + (1 - self.alpha) * hard_loss


# ----------------------------------------------------------------------------
# main() 함수: 폴드별 평가 진행 및 결과 엑셀 저장 (추가로 .h5 모델 지표 출력)
def main(args):
    os.makedirs(args.excel_dir, exist_ok=True)
    excel_path = os.path.join(args.excel_dir, "test_results.xlsx")
    df_dummy = pd.DataFrame()
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_dummy.to_excel(writer)

    date_folders = sorted([d for d in os.listdir(args.base_path) if os.path.isdir(os.path.join(args.base_path, d))])
    print(f"Total date folders found: {date_folders}")

    if not date_folders:
        raise ValueError("No date folders found in the base path. Please check the dataset directory.")

    results = pd.DataFrame(columns=[
        'Fold', 'Train Folders', 'Test Folder', 'Frame Confusion Matrix', 
        'Frame Accuracy', 'Video Confusion Matrix', 'Video Accuracy', 
        'Cut-off 10^2 CFU/ml', 'Frame F1 Score', 'Video F1 Score',
        'Row Weighted Scores', 'Row Total Weighted Scores', 'Model Size (KB)'
    ])
    results_summary = {
        'Frame Accuracy': [],
        'Video Accuracy': [],
        'Cut-off 10^2 CFU/ml': [],
        'Frame F1 Score': [],
        'Video F1 Score': [],
        'Row Weighted Scores': [],
        'Row Total Weighted Scores': []
    }

    cumulative_frame_conf_matrix = np.zeros((args.num_classes, args.num_classes))
    cumulative_video_conf_matrix = np.zeros((args.num_classes, args.num_classes))
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
            print(f"No data available for Fold {fold_index + 1}. Skipping this fold.")
            continue

        print(f"Fold {fold_index + 1}: Test data shape: {X_test.shape}, Test labels shape: {y_test.shape}")

        # -- PTQ 모델(TFLite) 사용: quantization_mode 인자에 따라 모델 파일 이름 구성 --
        # 예: Fold_1_ptq_model_fp16.tflite
        quant_mode_suffix = args.quantization_mode  # 'int8', 'fp16' 또는 'none'
        tflite_model_path = os.path.join(args.models_dir, f"Fold_{fold_index + 1}_ptq_model_{quant_mode_suffix}.tflite")
        if not os.path.exists(tflite_model_path):
            print(f"TFLite model file {tflite_model_path} does not exist. Skipping Fold {fold_index + 1}.")
            continue

        # --- .tflite 모델: 파일 크기 계산 ---
        size_bytes, size_kb, size_mb = get_model_size(tflite_model_path)
        print(f"TFLite model size: {size_bytes} bytes, {size_kb:.2f} KB, {size_mb:.2f} MB")

        # 프레임 단위 평가 (TFLite 모델 이용)
        frame_accuracy, frame_conf_matrix, row_weighted_scores, row_total_weighted_scores = evaluate_model_tflite(
            tflite_model_path, X_test, y_test, args.num_classes
        )
        cumulative_frame_conf_matrix += frame_conf_matrix

        print(f"Fold {fold_index + 1}: Frame Accuracy: {frame_accuracy * 100:.2f}%")
        
        # 영상 평가 (TFLite 모델 이용)
        video_conf_matrix, additional_no_stable_segments_videos = evaluate_videos(
            val_structure, tflite_model_path, args.img_roi, args.num_classes, args.equalize
        )
        no_stable_segments_videos_all_folds[-1][1].extend(additional_no_stable_segments_videos)
        cumulative_video_conf_matrix += video_conf_matrix
        video_accuracy = np.trace(video_conf_matrix) / np.sum(video_conf_matrix) if np.sum(video_conf_matrix) else 0

        print(f"Evaluating Fold {fold_index + 1}: test {test_folder}")
        print(f"Frame Confusion Matrix for Fold {fold_index + 1}:\n{frame_conf_matrix}")
        print(f"Frame Accuracy for Fold {fold_index + 1}: {frame_accuracy * 100:.2f}%\n")
        print(f"Row Weighted Scores for Fold {fold_index + 1}: {row_weighted_scores}")
        print(f"Row Total Weighted Scores for Fold {fold_index + 1}: {row_total_weighted_scores}\n")
        print(f"Video Confusion Matrix for Fold {fold_index + 1}:\n{video_conf_matrix}")
        print(f"Video Accuracy for Fold {fold_index + 1}: {video_accuracy * 100:.2f}%\n")

        frame_f1_scores = calculate_f1_scores(frame_conf_matrix)
        video_f1_scores = calculate_f1_scores(video_conf_matrix)

        print(f"Frame F1 Scores for Fold {fold_index + 1}: {frame_f1_scores}") 
        print(f"Video F1 Scores for Fold {fold_index + 1}: {video_f1_scores}") 

        if len(args.subfolders) == 5:
            accuracy_cutoff_revised = calculate_cutoff_accuracy(video_conf_matrix)
            print(f'Cut-off 10^2 CFU/ml Accuracy for Fold {fold_index + 1}: {accuracy_cutoff_revised * 100:.2f}%\n')
        else:
            accuracy_cutoff_revised = None

        test_folder_name = test_folder[0]

        fold_data = {
            'Fold': fold_index + 1,
            'Train Folders': ', '.join(train_folders),
            'Test Folder': test_folder_name,
            'Frame Confusion Matrix': frame_conf_matrix.tolist(),
            'Frame Accuracy': frame_accuracy,
            'Video Confusion Matrix': video_conf_matrix.tolist(),
            'Video Accuracy': video_accuracy,
            'Cut-off 10^2 CFU/ml': accuracy_cutoff_revised if accuracy_cutoff_revised is not None else "N/A",
            'Frame F1 Score': frame_f1_scores,
            'Video F1 Score': video_f1_scores,
            'Row Weighted Scores': [list(ws) for ws in row_weighted_scores],
            'Row Total Weighted Scores': row_total_weighted_scores,
            'Model Size (KB)': size_kb
        }

        fold_data_df = pd.DataFrame([fold_data])
        results = pd.concat([results, fold_data_df], ignore_index=True)
        
        results_summary['Frame Accuracy'].append(frame_accuracy)
        results_summary['Video Accuracy'].append(video_accuracy)
        results_summary['Cut-off 10^2 CFU/ml'].append(accuracy_cutoff_revised if accuracy_cutoff_revised is not None else 0)
        results_summary['Frame F1 Score'].append(frame_f1_scores)
        results_summary['Video F1 Score'].append(video_f1_scores)
        results_summary['Row Weighted Scores'].append(row_weighted_scores)
        results_summary['Row Total Weighted Scores'].append(row_total_weighted_scores)

        # --- 추가: .h5 모델에 대해 MACs, FLOPs, 파라미터 수, 모델 크기 계산 ---
        # H5 모델 파일은 "Fold_{fold_index+1}_model.h5"로 가정
        h5_model_path = os.path.join(args.models_dir, f"Fold_{fold_index + 1}_model.h5")
        if os.path.exists(h5_model_path):
            print(f"Computing metrics for h5 model: {h5_model_path}")
            try:
                # custom_object_scope로 사용자 정의 손실 함수를 등록합니다.
                with tf.keras.utils.custom_object_scope({'DistillationLoss': DistillationLoss}):
                    h5_model = tf.keras.models.load_model(h5_model_path)
                macs, flops, num_params, h5_model_size = get_h5_model_metrics(h5_model, (args.img_roi, args.img_roi, 1), h5_model_path)
                print(f"h5 Model Metrics:")
                print(f"  MACs: {macs:.2f}")
                print(f"  FLOPs: {flops:.2f}")
                print(f"  Number of parameters: {num_params}")
                print(f"  Model size: {h5_model_size/1024:.2f} KB")
            except Exception as e:
                print(f"Error computing h5 model metrics: {e}")
        else:
            print(f"No h5 model file found at {h5_model_path}")

    total_frame_accuracy = np.mean(results_summary['Frame Accuracy']) if results_summary['Frame Accuracy'] else 0
    total_video_accuracy = np.mean(results_summary['Video Accuracy']) if results_summary['Video Accuracy'] else 0
    total_cutoff_accuracy = np.mean(results_summary['Cut-off 10^2 CFU/ml']) if (len(args.subfolders) == 5 and results_summary['Cut-off 10^2 CFU/ml']) else None
    total_frame_f1_score = calculate_f1_scores(cumulative_frame_conf_matrix)
    total_video_f1_score = calculate_f1_scores(cumulative_video_conf_matrix)

    print("Total Frame Confusion Matrix:\n", np.round(cumulative_frame_conf_matrix).astype(int))
    print(f"Total Frame Accuracy: {total_frame_accuracy * 100:.2f}%")
    print(f"Total Frame F1 Score: {total_frame_f1_score}") 
    print("Total Video Confusion Matrix:\n", np.round(cumulative_video_conf_matrix).astype(int))
    print(f"Total Video Accuracy: {total_video_accuracy * 100:.2f}%")
    print(f"Total Video F1 Score: {total_video_f1_score}")

    if total_cutoff_accuracy is not None:
        print(f"Total Cut-off 10^2 CFU/ml Accuracy: {total_cutoff_accuracy * 100:.2f}%")

    total_results = pd.DataFrame({
        'Metric': [
            'Frame Accuracy', 'Video Accuracy', 'Cut-off 10^2 CFU/ml Accuracy', 
            'Average Frame F1 Score', 'Average Video F1 Score'
        ],
        'Value': [
            f"{total_frame_accuracy * 100:.2f}%",
            f"{total_video_accuracy * 100:.2f}%",
            f"{total_cutoff_accuracy * 100:.2f}%" if total_cutoff_accuracy is not None else "N/A",
            f"{total_frame_f1_score}",
            f"{total_video_f1_score}"
        ]
    })

    total_frame_conf_matrix_df = pd.DataFrame(
        cumulative_frame_conf_matrix, columns=[f"Pred_{i}" for i in range(cumulative_frame_conf_matrix.shape[1])]
    )
    total_frame_conf_matrix_df.index = [f"True_{i}" for i in range(total_frame_conf_matrix_df.shape[0])]
    total_video_conf_matrix_df = pd.DataFrame(
        cumulative_video_conf_matrix, columns=[f"Pred_{i}" for i in range(cumulative_video_conf_matrix.shape[1])]
    )
    total_video_conf_matrix_df.index = [f"True_{i}" for i in range(total_video_conf_matrix_df.shape[0])]

    results.to_excel(excel_path, index=False)
    with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a') as writer:
        total_results.to_excel(writer, sheet_name='Average Results', index=False)
        total_frame_conf_matrix_df.to_excel(writer, sheet_name='Avg Frame Confusion Matrix', index=True)
        total_video_conf_matrix_df.to_excel(writer, sheet_name='Avg Video Confusion Matrix', index=True)

        f1_frame_df = pd.DataFrame(total_frame_f1_score, columns=['F1 Score'])
        f1_frame_df.index = [f"Class_{i}" for i in range(len(total_frame_f1_score))]
        f1_frame_df.to_excel(writer, sheet_name='Frame F1 Scores', index=True)
        f1_video_df = pd.DataFrame(total_video_f1_score, columns=['F1 Score'])
        f1_video_df.index = [f"Class_{i}" for i in range(len(total_video_f1_score))]
        f1_video_df.to_excel(writer, sheet_name='Video F1 Scores', index=True)

    # 안정적인 세그먼트가 없는 비디오 목록 생성
    no_stable_segments_videos_df = pd.DataFrame(columns=['Fold', 'Date Folder', 'Subfolder', 'Video Name', 'Reason'])
    for fold_index, videos in no_stable_segments_videos_all_folds:
        for video in videos:
            date_folder, subfolder, video_name_folder, reason = video
            video_data = {
                'Fold': fold_index,
                'Date Folder': date_folder,
                'Subfolder': subfolder,
                'Video Name': video_name_folder,
                'Reason': reason
            }
            no_stable_segments_videos_df = pd.concat([no_stable_segments_videos_df, pd.DataFrame([video_data])], ignore_index=True)
    with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a') as writer:
        no_stable_segments_videos_df.to_excel(writer, sheet_name='No Stable Segments Videos', index=False)

    print(f"Results and averages have been saved to {excel_path}")
    print("\nVideos with No Stable Segments in Each Fold:")
    for fold_index, videos in no_stable_segments_videos_all_folds:
        if videos:
            print(f"Fold {fold_index}:")
            for video in videos:
                date_folder, subfolder, video_name_folder, reason = video
                print(f"  - Date Folder: {date_folder}, Subfolder: {subfolder}, Video: {video_name_folder}, Reason: {reason}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Script to test a PTQ TFLite CNN model on image data with histogram equalization.")
    parser.add_argument('--base_path', type=str, default='dataset/case_test/case16', help='Base directory for the dataset.')
    parser.add_argument('--models_dir', type=str, default='/home/jijang/ssd_data/projects/Bacteria/models/case_test/2025_case16_ptq_fp16', help='Directory where PTQ TFLite and h5 models are saved.')
    parser.add_argument('--excel_dir', type=str, default='excel/case_test/2025_case16_ptq_fp16',  help='Directory where excels are saved.')
    parser.add_argument('--subfolders', type=str, nargs='+', default=['0', '1', '2', '3', '4'], help='Subfolders to include as classes.')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes to predict.')
    parser.add_argument('--img_frame', type=int, default=900, help='Frame size of the images.')
    parser.add_argument('--img_roi', type=int, default=96, help='Region of interest size for each image.')
    parser.add_argument('--stability_threshold', type=int, default=350, help='Threshold for stability in video segmentation.')
    parser.add_argument('--buffer_size', type=int, default=2, help='Buffer size around detected peaks in stability analysis.')
    parser.add_argument('--equalize', action='store_true', help='Apply histogram equalization to images.')
    # quantization_mode 인자 추가: 'none', 'int8', 'fp16'
    parser.add_argument('--quantization_mode', type=str, default='fp16', choices=['none', 'int8', 'fp16'],
                        help='Quantization mode to choose the TFLite model file: none, int8, or fp16.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
