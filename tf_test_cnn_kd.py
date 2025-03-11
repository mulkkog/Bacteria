import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from tensorflow.keras.models import load_model
from collections import Counter
from scipy.signal import find_peaks
import pandas as pd
from PIL import Image

# GPU 설정
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

# FLOPs/MACs 계산을 위한 함수 (TF2.x)
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

def get_flops(model, batch_size=1):
    """
    주어진 Keras 모델의 총 FLOPs(부동소수점 연산 수)를 계산합니다.
    일반적으로 MACs는 FLOPs의 절반으로 추정할 수 있습니다.
    """
    concrete = tf.function(lambda x: model(x))
    input_shape = [batch_size] + list(model.input.shape[1:])
    concrete_func = concrete.get_concrete_function(tf.TensorSpec(input_shape, model.input.dtype))
    frozen_func = convert_variables_to_constants_v2(concrete_func)
    graph_def = frozen_func.graph.as_graph_def()
    
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name="")
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)
    return flops.total_float_ops

# 일반 모델을 불러옵니다.
def load_model_without_qat(model_path):
    model = load_model(model_path, compile=False)
    return model

def process_images(structure, img_roi, num_classes):
    X_data, y_data = [], []
    for date_folder, subfolders in structure.items():
        for subfolder, video_folders in subfolders.items():
            for video_folder, segments in video_folders.items():
                for segment_id, file_paths in segments.items():
                    if len(file_paths) <= 1:
                        continue
                    X = get_images(file_paths)
                    for _ in range(len(file_paths) - 1):
                        y_data.append(subfolder)
                    X_processed = abs(X - np.roll(X, -1, axis=0))
                    X_processed = np.delete(X_processed, -1, axis=0)
                    X_data.append(X_processed)
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
    return [i for i in range(total_frames) if i not in excluded_indices]

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
                            else:
                                no_stable_segments_videos.append((date_folder, subfolder, video_name_folder, "No stable segments"))
                                print(f"No stable segments found in {video_name_folder}")
    return structure, no_stable_segments_videos

def load_tiff_images(folder_path, file_names):
    images = []
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        with Image.open(file_path) as img:
            images.append(np.array(img))
    return np.array(images) / 255.0

def load_yuv_images(folder_path, file_names, image_size=(128, 128)):
    images = []
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'rb') as file:
            image = np.frombuffer(file.read(), dtype=np.uint8)
            image = image.reshape(image_size)
            images.append(image)
    return np.array(images) / 255.0

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
    return find_all_stable_segments(summed_diffs, peaks, buffer_size)

def get_images(file_paths):
    images = []
    for file_path in file_paths:
        with open(file_path, 'rb') as image_file:
            image = np.frombuffer(image_file.read(128 * 128), dtype=np.uint8).reshape((128, 128))
            image = np.array(image) / 255.0
            images.append(image)
    return images

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)
    accuracy = np.mean(y_pred == y_test)
    conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3, 4])
    weights = np.array([0, 10, 100, 1000, 10000])
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
        if top2_index == top1_index - 1 or top2_index == top1_index + 1:
            top2_scores = top2_value * weights[top2_index] / total_samples
            total_weighted_score = top1_scores + top2_scores
        else:
            total_weighted_score = top1_scores
        row_weighted_scores.append([row[i] * weights[i] / total_samples for i in range(len(row))])
        row_total_weighted_scores.append(total_weighted_score)
    return accuracy, conf_matrix, row_weighted_scores, row_total_weighted_scores

def preprocess_data(test_path, img_roi):
    X = get_images(test_path)
    X_tests = abs(X - np.roll(X, -1, axis=0))
    X_tests = np.delete(X_tests, -1, axis=0)
    X_tests = np.expand_dims(X_tests, axis=-1)
    X_tests = np.asarray(X_tests).astype(np.float32)
    return X_tests[:, :img_roi, :img_roi, :]

def preprocess_data_for_tflite(images, input_details):
    """
    INT8 모델이면 quantization 파라미터(scale, zero_point)를 이용해 변환하고,
    FP16 또는 FP32 모델이면 별도 변환 없이 (필요시 dtype 캐스팅) 반환합니다.
    """
    dtype = input_details[0]['dtype']
    if dtype == np.int8:
        scale, zero_point = input_details[0]['quantization']
        images = images / scale + zero_point
        images = np.clip(images, -128, 127)
        return images.astype(np.int8)
    else:
        if dtype == np.float16:
            return images.astype(np.float16)
        else:
            return images.astype(np.float32)

def postprocess_output_from_tflite(output_data, output_details):
    """
    INT8 모델이면 quantization 정보를 적용하여 후처리하고,
    FP16/FP32 모델이면 그대로 반환합니다.
    """
    dtype = output_details[0]['dtype']
    if dtype == np.int8:
        scale, zero_point = output_details[0]['quantization']
        return (output_data.astype(np.float32) - zero_point) * scale
    else:
        if dtype == np.float16:
            return output_data.astype(np.float16)
        else:
            return output_data.astype(np.float32)

def get_tflite_quantization_type(tflite_model_path):
    """
    TFLite 모델의 입력 텐서를 확인하여 INT8인지, FP16(또는 FP32)인지 구분합니다.
    """
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    dtype = input_details[0]['dtype']
    if dtype == np.int8:
        return "INT8"
    elif dtype == np.float16:
        return "FP16"
    else:
        return "FP32 or Unknown"

def model_inference(X_test, tflite_model_path):
    """
    TFLite 모델을 이용해 전체 배치(X_test)를 한 번에 inference 합니다.
    모델의 입력 dtype에 따라 전처리 후 추론하고, 출력도 후처리합니다.
    """
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 배치 크기에 맞게 입력 텐서 크기 변경 (resize)
    batch_size = X_test.shape[0]
    new_input_shape = [batch_size] + list(input_details[0]['shape'][1:])
    interpreter.resize_tensor_input(input_details[0]['index'], new_input_shape)
    interpreter.allocate_tensors()

    # 입력 데이터 전처리
    if input_details[0]['dtype'] == np.int8:
        input_scale, input_zero_point = input_details[0]['quantization']
        X_test_quant = (X_test / input_scale + input_zero_point).astype(np.int8)
    else:
        X_test_quant = X_test.astype(input_details[0]['dtype'])

    interpreter.set_tensor(input_details[0]['index'], X_test_quant)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    if output_details[0]['dtype'] == np.int8:
        output_scale, output_zero_point = output_details[0]['quantization']
        output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale

    return np.argmax(output_data, axis=1)

def find_mode(lst):
    lst = lst.reshape(-1)
    return Counter(lst).most_common(1)[0][0]

def evaluate_videos(structure, tflite_model_path, img_roi, num_classes):
    confusion_matrix_video = np.zeros((num_classes, num_classes), dtype=int)
    video_results, actual_classes = {}, {}
    insufficient_frame_videos = set()
    for date_folder, subfolders in structure.items():
        for subfolder, video_folders in subfolders.items():
            for video_folder, segments in video_folders.items():
                insufficient_frames_count = 0
                for segment_id, file_paths in segments.items():
                    if (file_paths is None) or (len(file_paths) <= 1):
                        insufficient_frames_count += 1
                        continue
                    images = preprocess_data(file_paths, img_roi)
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
        confusion_matrix_video[int(subfolder), final_class] += 1
    return confusion_matrix_video, list(insufficient_frame_videos)

def calculate_cutoff_accuracy(conf_matrix):
    tp_top_left = np.sum(conf_matrix[0:2, 0:2])
    tp_bottom_right = np.sum(conf_matrix[2:, 2:])
    total_true_positives = tp_top_left + tp_bottom_right
    total_samples = np.sum(conf_matrix)
    return total_true_positives / total_samples if total_samples else 0

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

def main(args):
    os.makedirs(args.excel_dir, exist_ok=True)
    excel_path = os.path.join(args.excel_dir, "test_results.xlsx")
    df_dummy = pd.DataFrame()
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_dummy.to_excel(writer)
    date_folders = sorted([d for d in os.listdir(args.base_path) if os.path.isdir(os.path.join(args.base_path, d))])
    kf = KFold(n_splits=len(date_folders))
    results = pd.DataFrame(columns=[
        'Fold', 'Train Folders', 'Test Folder', 'Frame Confusion Matrix', 
        'Frame Accuracy', 'Video Confusion Matrix', 'Video Accuracy', 
        'Cut-off 10^2 CFU/ml'
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
    for fold_index, (train_index, test_index) in enumerate(kf.split(date_folders)):
        print(f"\n--- Processing Fold {fold_index + 1} ---")
        # 모델 파일 이름에서 'qat' 등 제거 후 불러오기
        model_path = os.path.join(args.student_model_dir, f"Fold_{fold_index + 1}_model.h5")
        model = load_model_without_qat(model_path)
        try:
            total_flops = get_flops(model, batch_size=1)
            total_macs = total_flops / 2
            print(f"Fold {fold_index + 1}: Model FLOPs: {total_flops}")
            print(f"Fold {fold_index + 1}: Model MACs (approx.): {total_macs}")
        except Exception as e:
            print(f"Fold {fold_index + 1}: Error computing FLOPs/MACs: {e}")
        trainable_params = model.count_params()
        print(f"Fold {fold_index + 1}: Number of trainable parameters: {trainable_params}")
        model_size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"Fold {fold_index + 1}: .h5 Model size: {model_size:.2f} MB")
        test_folder = [date_folders[test_index[0]]]
        test_structure, no_stable_segments_videos = get_temporary_structure_with_stable_segments(
            args.base_path, test_folder, args.subfolders, args.stability_threshold, args.buffer_size
        )
        no_stable_segments_videos_all_folds.append((fold_index + 1, no_stable_segments_videos))
        X_test, y_test = process_images(test_structure, args.img_roi, args.num_classes)
        frame_accuracy, frame_conf_matrix, row_weighted_scores, row_total_weighted_scores = evaluate_model(model, X_test, y_test)
        
        # TFLite 모델 파일명 (FP32로 저장)
        tflite_model_filename = f"Fold_{fold_index + 1}_model_fp32.tflite"
        tflite_model_path = os.path.join(args.student_model_dir, tflite_model_filename)
        if not os.path.exists(tflite_model_path):
            print(f"TFLite model not found at {tflite_model_path}. Converting to FP32 TFLite model...")
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            # FP32 변환 시 추가적인 quantization 옵션 없이 변환
            tflite_model = converter.convert()
            with open(tflite_model_path, "wb") as f:
                f.write(tflite_model)
            print("TFLite FP32 model saved.")
        else:
            tflite_model_size = os.path.getsize(tflite_model_path) / (1024 * 1024)
            print(f"Fold {fold_index + 1}: TFLite Model size: {tflite_model_size:.2f} MB")
            print("Using existing FP32 TFLite model.")
        
        video_conf_matrix, additional_no_stable_segments_videos = evaluate_videos(
            test_structure, tflite_model_path, args.img_roi, args.num_classes
        )
        no_stable_segments_videos_all_folds[-1][1].extend(additional_no_stable_segments_videos)
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
        train_folders = [date_folders[i] for i in train_index]
        test_folder_name = date_folders[test_index[0]]
        fold_data = {
            'Fold': fold_index + 1,
            'Train Folders': ', '.join(train_folders),
            'Test Folder': test_folder_name,
            'Frame Confusion Matrix': frame_conf_matrix.tolist(),
            'Frame Accuracy': frame_accuracy,
            'Video Confusion Matrix': video_conf_matrix.tolist(),
            'Video Accuracy': video_accuracy,
            'Frame F1 Score': frame_f1_scores,
            'Video F1 Score': video_f1_scores,
            'Row Weighted Scores': [list(ws) for ws in row_weighted_scores],
            'Row Total Weighted Scores': row_total_weighted_scores
        }
        if len(args.subfolders) == 5:
            fold_data['Cut-off 10^2 CFU/ml'] = accuracy_cutoff_revised
        fold_data_df = pd.DataFrame([fold_data])
        results = pd.concat([results, fold_data_df], ignore_index=True)
        results_summary['Frame Accuracy'].append(frame_accuracy)
        results_summary['Video Accuracy'].append(video_accuracy)
        results_summary['Frame F1 Score'].append(frame_f1_scores)
        results_summary['Video F1 Score'].append(video_f1_scores)
        results_summary['Row Weighted Scores'].append(row_weighted_scores)
        results_summary['Row Total Weighted Scores'].append(row_total_weighted_scores)
        if len(args.subfolders) == 5:
            results_summary['Cut-off 10^2 CFU/ml'].append(accuracy_cutoff_revised)
        cumulative_frame_conf_matrix += np.array(frame_conf_matrix)
        cumulative_video_conf_matrix += np.array(video_conf_matrix)
    total_frame_conf_matrix = cumulative_frame_conf_matrix
    total_video_conf_matrix = cumulative_video_conf_matrix
    total_frame_accuracy = sum(results_summary['Frame Accuracy']) / len(results_summary['Frame Accuracy'])
    total_video_accuracy = sum(results_summary['Video Accuracy']) / len(results_summary['Video Accuracy'])
    total_cutoff_accuracy = (
        sum(results_summary['Cut-off 10^2 CFU/ml']) / len(results_summary['Cut-off 10^2 CFU/ml'])
        if len(args.subfolders) == 5 else None
    )
    total_frame_f1_score = calculate_f1_scores(total_frame_conf_matrix)
    total_video_f1_score = calculate_f1_scores(total_video_conf_matrix)
    combined_row_weighted_scores = np.sum([np.array(scores) for scores in results_summary['Row Weighted Scores']], axis=0)
    combined_row_total_weighted_scores = np.sum([np.array(scores) for scores in results_summary['Row Total Weighted Scores']], axis=0)
    print("Total Frame Confusion Matrix:\n", np.round(total_frame_conf_matrix).astype(int))
    print(f"Total Frame Accuracy: {total_frame_accuracy * 100:.2f}%")
    print(f"Total Frame F1 Score: {total_frame_f1_score}") 
    print(f"Combined Row Weighted Scores: {combined_row_weighted_scores.tolist()}")
    print(f"Combined Row Total Weighted Scores: {combined_row_total_weighted_scores.tolist()}")
    print("Total Video Confusion Matrix:\n", np.round(total_video_conf_matrix).astype(int))
    print(f"Total Video Accuracy: {total_video_accuracy * 100:.2f}%")
    print(f"Total Video F1 Score: {total_video_f1_score}")
    if total_cutoff_accuracy is not None:
        print(f"Total Cut-off 10^2 CFU/ml Accuracy: {total_cutoff_accuracy * 100:.2f}%")
    total_results = pd.DataFrame({
        'Metric': [
            'Frame Accuracy', 'Video Accuracy', 'Cut-off 10^2 CFU/ml Accuracy', 
            'Average Frame F1 Score', 'Average Video F1 Score', 'Combined Row Weighted Scores', 'Combined Row Total Weighted Scores'
        ],
        'Value': [
            f"{total_frame_accuracy * 100:.2f}%",
            f"{total_video_accuracy * 100:.2f}%",
            f"{total_cutoff_accuracy * 100:.2f}%" if total_cutoff_accuracy is not None else "N/A",
            f"{total_frame_f1_score}",
            f"{total_video_f1_score}",
            f"{combined_row_weighted_scores.tolist()}",
            f"{combined_row_total_weighted_scores.tolist()}"
        ]
    })
    total_frame_conf_matrix_df = pd.DataFrame(
        total_frame_conf_matrix, columns=[f"Pred_{i}" for i in range(total_frame_conf_matrix.shape[1])]
    )
    total_frame_conf_matrix_df.index = [f"True_{i}" for i in range(total_frame_conf_matrix_df.shape[0])]
    total_video_conf_matrix_df = pd.DataFrame(
        total_video_conf_matrix, columns=[f"Pred_{i}" for i in range(total_video_conf_matrix.shape[1])]
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
        combined_weighted_scores_df = pd.DataFrame(combined_row_weighted_scores, columns=[f"Weight_{i}" for i in range(combined_row_weighted_scores.shape[1])])
        combined_weighted_scores_df.index = [f"Class_{i}" for i in range(combined_row_weighted_scores.shape[0])]
        combined_weighted_scores_df.to_excel(writer, sheet_name='Row Weighted Scores', index=True)
        combined_total_weighted_scores_df = pd.DataFrame(combined_row_total_weighted_scores, columns=['Total Weighted Score'])
        combined_total_weighted_scores_df.index = [f"Class_{i}" for i in range(combined_row_total_weighted_scores.shape[0])]
        combined_total_weighted_scores_df.to_excel(writer, sheet_name='Row Total Weighted Scores', index=True)
        no_stable_segments_videos_df = pd.DataFrame(columns=['Fold', 'Date Folder', 'Subfolder', 'Video Name', 'Reason'])
        list_video_data = []
        for fold_index, videos in no_stable_segments_videos_all_folds:
            for date_folder, subfolder, video_name_folder, reason in videos:
                video_data = {
                    'Fold': fold_index,
                    'Date Folder': date_folder,
                    'Subfolder': subfolder,
                    'Video Name': video_name_folder,
                    'Reason': reason
                }
                list_video_data.append(video_data)
        if list_video_data:
            no_stable_segments_videos_df = pd.concat([no_stable_segments_videos_df, pd.DataFrame(list_video_data)], ignore_index=True)
        no_stable_segments_videos_df.to_excel(writer, sheet_name='No Stable Segments Videos', index=False)
    print(f"Results and averages have been saved to {excel_path}")
    print("\nVideos with No Stable Segments in Each Fold:")
    for fold_index, videos in no_stable_segments_videos_all_folds:
        if videos:
            print(f"Fold {fold_index}:")
            for date_folder, subfolder, video_name_folder, reason in videos:
                print(f"  - Date Folder: {date_folder}, Subfolder: {subfolder}, Video: {video_name_folder}, Reason: {reason}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Script to evaluate a convolutional neural network on image data.")
    parser.add_argument('--base_path', type=str, default='/home/jijang/ssd_data/projects/Bacteria/dataset/case_test/case16', help='Base directory for the dataset.')
    parser.add_argument('--student_model_dir', type=str, default='/home/jijang/ssd_data/projects/Bacteria/models/case_test/2025_case16_kd', help='Directory where student models are saved.')
    parser.add_argument('--excel_dir', type=str, default='/home/jijang/ssd_data/projects/Bacteria/excel/case_test/2025_case16_kd',  help='Directory where excels are saved.')
    parser.add_argument('--subfolders', type=str, nargs='+', default=['0', '1', '2', '3', '4'], help='Subfolders to include as classes.')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes to predict.')
    parser.add_argument('--img_frame', type=int, default=900, help='Frame size of the images.')
    parser.add_argument('--img_roi', type=int, default=96, help='Region of interest size for each image.')
    parser.add_argument('--stability_threshold', type=int, default=350, help='Threshold for stability in video segmentation.')
    parser.add_argument('--buffer_size', type=int, default=0, help='Buffer size around detected peaks in stability analysis.')
    # quantization 옵션 없이 FP32로 저장하므로, 여기서는 사용하지 않습니다.
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
