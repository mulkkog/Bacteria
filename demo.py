import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from collections import Counter
from scipy.signal import find_peaks
import pandas as pd
from PIL import Image


def process_images(structure, img_roi, num_classes, class_label):
    X_data = []
    y_data = []
    for video_folder, segments in structure.items():
        for segment_id, file_paths in segments.items():
            if len(file_paths) <= 1:
                continue
            X = get_images(file_paths)
            for _ in range(len(file_paths) - 1):
                y_data.append(class_label)  # 클래스 레이블을 사용자가 지정한 값으로 설정
            X_processed = abs(X - np.roll(X, -1, axis=0))
            X_processed = np.delete(X_processed, -1, axis=0)
            X_data.append(X_processed)
    X_data = np.vstack(X_data).astype(np.float32)
    X_data = np.expand_dims(X_data, axis=-1)
    y_data = to_categorical(y_data, num_classes)  # y_data는 class_label로 채워짐
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

def get_temporary_structure_with_stable_segments(base_path, stability_threshold, buffer_size):
    structure = {}
    no_stable_segments_videos = []  # stable segment가 없는 비디오를 저장할 리스트

    for video_name_folder in os.listdir(base_path):
        video_folder_path = os.path.join(base_path, video_name_folder)
        if os.path.isdir(video_folder_path):
            video_files = [f for f in os.listdir(video_folder_path) if f.endswith('.yuv') or f.endswith('.tiff')]
            video_files_sorted = sorted(video_files)

            stable_segments = process_video_files(video_folder_path, video_files_sorted, stability_threshold, buffer_size)

            if stable_segments:
                structure[video_name_folder] = {}
                start = stable_segments[0]
                segment_id = 0
                for i in range(1, len(stable_segments)):
                    if stable_segments[i] != stable_segments[i - 1] + 1:
                        end = stable_segments[i - 1] + 1
                        segment_files = [os.path.join(video_folder_path, video_files_sorted[j]) for j in range(start, end)]
                        structure[video_name_folder][f"segment_{segment_id}"] = segment_files
                        segment_id += 1
                        start = stable_segments[i]
                # Add the last segment
                end = stable_segments[-1] + 1
                segment_files = [os.path.join(video_folder_path, video_files_sorted[j]) for j in range(start, end)]
                structure[video_name_folder][f"segment_{segment_id}"] = segment_files
            else:
                no_stable_segments_videos.append((video_name_folder, "No stable segments"))  # stable segment가 없는 비디오 추가
                print(f"No stable segments found in {video_name_folder}")

    return structure, no_stable_segments_videos  # stable segment가 없는 비디오 목록도 반환


# TIFF 이미지 로드 함수
def load_tiff_images(folder_path, file_names):
    images = []
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        with Image.open(file_path) as img:
            images.append(np.array(img))
    return np.array(images) / 255.0


# YUV 이미지 로드 함수
def load_yuv_images(folder_path, file_names, image_size=(128, 128)):
    images = []
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'rb') as file:
            image = np.frombuffer(file.read(), dtype=np.uint8)
            image = image.reshape(image_size)
            images.append(image)
    return np.array(images) / 255.0


# 비디오 파일 처리 함수
def process_video_files(folder_path, file_names, stability_threshold, buffer_size):
    if not file_names:
        print("The folder is empty. No files to process.")
        return []
        
    # Determine file format from the first file
    if file_names[0].endswith('.yuv'):
        images = load_yuv_images(folder_path, file_names)
    elif file_names[0].endswith('.tiff'):
        images = load_tiff_images(folder_path, file_names)
    else:
        raise ValueError("Unsupported file format")

    # Calculate differences between images
    image_diffs = np.abs(images[:-1] - images[1:])
    summed_diffs = image_diffs.sum(axis=(1, 2))

    # Find peaks in the 1D array of summed differences
    peaks, _ = find_peaks(summed_diffs, height=stability_threshold)

    # Find top stable segments
    stable_segments = find_all_stable_segments(summed_diffs, peaks, buffer_size)
    return stable_segments

def get_images(file_paths):
    images = []
    for file_path in file_paths:
        with open(file_path, 'rb') as image_file:
            image = np.frombuffer(image_file.read(128 * 128), dtype=np.uint8).reshape((128, 128))
            image = np.array(image) / 255.0
            images.append(image)
    return images


def evaluate_videos(structure, tflite_model_path, img_roi, num_classes, class_label):
    frame_conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    video_conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

    video_results = {}
    insufficient_frame_videos = set()  # Insufficient frames 비디오를 저장할 집합

    for video_folder, segments in structure.items():
        insufficient_frames_count = 0
        for segment_id, file_paths in segments.items():
            if (file_paths is None) or (len(file_paths) <= 1):  # file_paths가 None인지도 확인
                insufficient_frames_count += 1
                continue

            images = preprocess_data(file_paths, img_roi)  # (299, 96, 96, 1)
            result = model_inference(images, tflite_model_path)
            
            # 각 프레임에 대한 혼동 행렬 업데이트
            for frame_pred in result:
                frame_conf_matrix[class_label, frame_pred] += 1

            most_common_result = find_mode(result)

            if video_folder not in video_results:
                video_results[video_folder] = []
            video_results[video_folder].append(most_common_result)

        # 모든 세그먼트가 insufficient frames 인 경우에만 추가
        if insufficient_frames_count == len(segments):
            insufficient_frame_videos.add((video_folder, "Insufficient frames"))

    # 비디오 단위의 결과를 이용한 혼동 행렬 업데이트
    for video_folder, results in video_results.items():
        final_class = find_mode(np.array(results))
        video_conf_matrix[class_label, final_class] += 1  # args로 받은 class_label 사용

    # 프레임별 가중치 계산
    frame_weighted_scores, frame_total_weighted_scores = calculate_weighted_scores(frame_conf_matrix)

    # 비디오별 가중치 계산
    video_weighted_scores, video_total_weighted_scores = calculate_weighted_scores(video_conf_matrix)

    return (frame_conf_matrix, video_conf_matrix, frame_weighted_scores, frame_total_weighted_scores, 
            video_weighted_scores, video_total_weighted_scores, list(insufficient_frame_videos))  


def preprocess_data(test_path, img_roi):
    X_test = []

    X = get_images(test_path)
    X = np.array(X)
    X_tests = abs(X - np.roll(X, -1, axis=0))
    X_tests = np.delete(X_tests, -1, axis=0)

    X_test = np.expand_dims(X_tests, axis=-1)
    X_test = np.asarray(X_test).astype(np.float32)
    X_test = X_test[:, :img_roi, :img_roi, :]
    return X_test


def model_inference(X_test, tflite_model_path):
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(tflite_model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = X_test
    output_data = []

    for i in range(input_data.shape[0]):
        single_input = np.expand_dims(input_data[i], axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], single_input)
        interpreter.invoke()
        single_output = interpreter.get_tensor(output_details[0]['index'])
        single_output = np.argmax(single_output, axis=1)
        output_data.append(single_output)

    output_data = np.array(output_data)
    return output_data


def find_mode(lst):
    lst = lst.reshape(-1)
    counter = Counter(lst)
    most_common = counter.most_common(1)
    return most_common[0][0]


def calculate_weighted_scores(conf_matrix):
    weights = np.array([0, 10, 100, 1000, 10000])
    row_weighted_scores = []
    row_total_weighted_scores = []

    for row in conf_matrix:
        total_samples = np.sum(row)
        if total_samples > 0:
            class_percentages = (row / total_samples) * 100
            weighted_scores = weights * (class_percentages / 100)
            total_weighted_score = np.sum(weighted_scores)
        else:
            weighted_scores = np.zeros_like(weights)
            total_weighted_score = 0

        row_weighted_scores.append(weighted_scores)
        row_total_weighted_scores.append(total_weighted_score)

    return row_weighted_scores, row_total_weighted_scores


def main(args):
    os.makedirs(args.excel_dir, exist_ok=True)
    excel_path = os.path.join(args.excel_dir, "test_results_demo.xlsx")
    df_dummy = pd.DataFrame()
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_dummy.to_excel(writer)

    # Test on the subfolder (base_path)
    test_structure, no_stable_segments_videos = get_temporary_structure_with_stable_segments(
        args.base_path, args.stability_threshold, args.buffer_size
    )

    # 여기서 class_label은 사용자가 지정하는 레이블입니다.
    X_test, y_test = process_images(test_structure, args.img_roi, args.num_classes, class_label=args.class_label)

    # Evaluate videos using the TFLite model
    (frame_conf_matrix, video_conf_matrix, frame_weighted_scores, frame_total_weighted_scores,
     video_weighted_scores, video_total_weighted_scores, additional_no_stable_segments_videos) = evaluate_videos(
        test_structure, args.tflite_model_path, args.img_roi, args.num_classes, class_label=args.class_label
    )
    no_stable_segments_videos.extend(additional_no_stable_segments_videos)

    frame_accuracy = np.trace(frame_conf_matrix) / np.sum(frame_conf_matrix) if np.sum(frame_conf_matrix) else 0
    video_accuracy = np.trace(video_conf_matrix) / np.sum(video_conf_matrix) if np.sum(video_conf_matrix) else 0

    # Print evaluation results
    print(f"Evaluating test on base path {args.base_path}")
    print(f"Frame Confusion Matrix:\n{frame_conf_matrix}")
    print(f"Frame Accuracy: {frame_accuracy * 100:.2f}%\n")
    print(f"Video Confusion Matrix:\n{video_conf_matrix}")
    print(f"Video Accuracy: {video_accuracy * 100:.2f}%\n")
    print(f"Frame Total Weighted Scores: {frame_total_weighted_scores}\n")
    print(f"Video Total Weighted Scores: {video_total_weighted_scores}\n")

    # Save to Excel 
    results = pd.DataFrame({
        'Metric': ['Frame Accuracy', 'Video Accuracy'],
        'Value': [f"{frame_accuracy * 100:.2f}%", f"{video_accuracy * 100:.2f}%"]
    })

    results.to_excel(excel_path, index=False)

    # Save confusion matrices and weighted scores to Excel
    with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a') as writer:
        frame_conf_matrix_df = pd.DataFrame(frame_conf_matrix, index=[f"True_{i}" for i in range(args.num_classes)],
                                            columns=[f"Pred_{i}" for i in range(args.num_classes)])
        frame_conf_matrix_df.to_excel(writer, sheet_name='Frame Confusion Matrix')

        video_conf_matrix_df = pd.DataFrame(video_conf_matrix, index=[f"True_{i}" for i in range(args.num_classes)],
                                            columns=[f"Pred_{i}" for i in range(args.num_classes)])
        video_conf_matrix_df.to_excel(writer, sheet_name='Video Confusion Matrix')

        frame_weighted_scores_df = pd.DataFrame(frame_weighted_scores, index=[f"Class_{i}" for i in range(args.num_classes)],
                                                columns=[f"Weight_{i}" for i in range(len(frame_weighted_scores[0]))])
        frame_weighted_scores_df.to_excel(writer, sheet_name='Frame Weighted Scores')

        video_weighted_scores_df = pd.DataFrame(video_weighted_scores, index=[f"Class_{i}" for i in range(args.num_classes)],
                                                columns=[f"Weight_{i}" for i in range(len(video_weighted_scores[0]))])
        video_weighted_scores_df.to_excel(writer, sheet_name='Video Weighted Scores')

    print(f"Results have been saved to {excel_path}")
    print("\nVideos with No Stable Segments:")
    for video_name_folder, reason in no_stable_segments_videos:
        print(f"  - Video: {video_name_folder}, Reason: {reason}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Demo script to evaluate a TFLite model on video data.")
    parser.add_argument('--base_path', type=str, default='/home/jijang/projects/Bacteria/dataset/case_test/case1/0527/1', help='Base directory for the dataset.')
    parser.add_argument('--tflite_model_path', type=str, default='/home/jijang/projects/Bacteria/models/case_test/240802_case4_tf/Fold_1_HTCNN.tflite', help='Path to the TFLite model file.')
    parser.add_argument('--excel_dir', type=str, default='/home/jijang/projects/Bacteria/excel/case_test/240820_case4_tf', help='Directory where excels are saved.')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes to predict.')
    parser.add_argument('--img_roi', type=int, default=96, help='Region of interest size for each image.')
    parser.add_argument('--stability_threshold', type=int, default=350, help='Threshold for stability in video segmentation.')
    parser.add_argument('--buffer_size', type=int, default=0, help='Buffer size around detected peaks in stability analysis.')
    parser.add_argument('--class_label', type=int, default=1, help='Class label to assign to all data in this run.')  # 새로운 인자 추가
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
