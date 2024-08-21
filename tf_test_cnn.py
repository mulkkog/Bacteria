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
from tensorflow.keras import backend as K
from PIL import Image

# GPU 설정
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1 "
 
def process_images(structure, img_roi, num_classes):
    X_data = []
    y_data = []
    for date_folder, subfolders in structure.items():
        for subfolder, video_folders in subfolders.items():
            for video_folder, segments in video_folders.items():
                for segment_id, file_paths in segments.items():
                    if len(file_paths) <= 1:
                        continue
                    X = get_images(file_paths)
                    for _ in range(len(file_paths) - 1):
                        y_data.append(subfolder)  # Assuming subfolder name is the class label
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
    valid_frames = [i for i in range(total_frames) if i not in excluded_indices]
    return valid_frames

def get_temporary_structure_with_stable_segments(base_path, date_folders, subfolders, stability_threshold, buffer_size):
    structure = {}
    no_stable_segments_videos = []  # stable segment가 없는 비디오를 저장할 리스트

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
                                # Add the last segment
                                end = stable_segments[-1] + 1
                                segment_files = [os.path.join(video_folder_path, video_files_sorted[j]) for j in range(start, end)]
                                structure[date_folder][subfolder][video_name_folder][f"segment_{segment_id}"] = segment_files
                            else:
                                no_stable_segments_videos.append((date_folder, subfolder, video_name_folder, "No stable segments"))  # stable segment가 없는 비디오 추가
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


def evaluate_model(model, X_test, y_test):
    # Predict the class probabilities for the test set
    y_pred = model.predict(X_test)
    
    # Convert predictions and true labels to class indices
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)
    
    # Calculate the accuracy
    accuracy = np.mean(y_pred == y_test)
    
    # Compute the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3, 4])
    
    # Define weights for each class
    weights = np.array([0, 10, 100, 1000, 10000])
    
    # Calculate the class distribution percentages and weighted scores per row
    row_weighted_scores = []
    row_total_weighted_scores = []

    for row in conf_matrix:
        total_samples = np.sum(row)
        class_percentages = (row / total_samples) * 100
        weighted_scores = weights * (class_percentages / 100)
        total_weighted_score = np.sum(weighted_scores)

        row_weighted_scores.append(weighted_scores)
        row_total_weighted_scores.append(total_weighted_score)
    
    return accuracy, conf_matrix, row_weighted_scores, row_total_weighted_scores


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


def evaluate_videos(structure, tflite_model_path, img_roi, num_classes):
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    video_results = {}
    actual_classes = {}
    insufficient_frame_videos = set()  # Insufficient frames 비디오를 저장할 집합

    for date_folder, subfolders in structure.items():
        for subfolder, video_folders in subfolders.items():
            for video_folder, segments in video_folders.items():
                insufficient_frames_count = 0
                for segment_id, file_paths in segments.items():
                    if (file_paths is None) or (len(file_paths) <= 1):  # file_paths가 None인지도 확인
                        insufficient_frames_count += 1
                        continue

                    images = preprocess_data(file_paths, img_roi)  # (299, 96, 96, 1)
                    result = model_inference(images, tflite_model_path)
                    most_common_result = find_mode(result)

                    if video_folder not in video_results:
                        video_results[video_folder] = []
                        actual_classes[video_folder] = subfolder  # 실제 클래스 ID 저장
                    video_results[video_folder].append(most_common_result)

                # 모든 세그먼트가 insufficient frames 인 경우에만 추가
                if insufficient_frames_count == len(segments):
                    insufficient_frame_videos.add((date_folder, subfolder, video_folder, "Insufficient frames"))

    for video_folder, results in video_results.items():
        final_class = find_mode(np.array(results))
        subfolder = actual_classes[video_folder]  # 실제 클래스 ID 사용
        confusion_matrix[int(subfolder), final_class] += 1

    return confusion_matrix, list(insufficient_frame_videos)  # Insufficient frames 비디오 목록도 반환


def calculate_cutoff_accuracy(conf_matrix):
    # 상단 왼쪽 2x2 영역의 정확한 예측 수
    tp_top_left = np.sum(conf_matrix[0:2, 0:2])
    # 하단 오른쪽 3x3 영역의 정확한 예측 수
    tp_bottom_right = np.sum(conf_matrix[2:, 2:])
    # 총 정확한 예측 수
    total_true_positives = tp_top_left + tp_bottom_right
    # 전체 샘플 수
    total_samples = np.sum(conf_matrix)

    # 정확도 계산
    accuracy = total_true_positives / total_samples if total_samples else 0
    return accuracy


def calculate_f1_scores(conf_matrix):
    # Calculate F1 scores for each class
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

    date_folders = [d for d in os.listdir(args.base_path) if os.path.isdir(os.path.join(args.base_path, d))]
    kf = KFold(n_splits=len(date_folders))

    # Initialize the results DataFrame
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
        'Row Weighted Scores': [],  # Changed this to store row-wise weighted scores
        'Row Total Weighted Scores': []  # Changed this to store row-wise total weighted scores
    }

    # Initialize matrices for cumulative confusion matrix accumulation
    cumulative_frame_conf_matrix = np.zeros((args.num_classes, args.num_classes))
    cumulative_video_conf_matrix = np.zeros((args.num_classes, args.num_classes))
    no_stable_segments_videos_all_folds = []  # List for videos with no stable segments

    for fold_index, (train_index, test_index) in enumerate(kf.split(date_folders)):
        model_path = os.path.join(args.models_dir, f"Fold_{fold_index + 1}_HTCNN.h5")
        model = load_model(model_path)
        test_folder = [date_folders[test_index[0]]]
        test_structure, no_stable_segments_videos = get_temporary_structure_with_stable_segments(
            args.base_path, test_folder, args.subfolders, args.stability_threshold, args.buffer_size
        )
        no_stable_segments_videos_all_folds.append((fold_index + 1, no_stable_segments_videos))
        X_test, y_test = process_images(test_structure, args.img_roi, args.num_classes)

        # Evaluate the model
        frame_accuracy, frame_conf_matrix, row_weighted_scores, row_total_weighted_scores = evaluate_model(model, X_test, y_test)

        # Convert model to TFLite format
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        converter._experimental_lower_tensor_list_ops = False
        tflite_model = converter.convert()
        tflite_model_path = os.path.join(args.models_dir, f"Fold_{fold_index + 1}_HTCNN.tflite")
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model)
        
        # Evaluate videos using the TFLite model
        video_conf_matrix, additional_no_stable_segments_videos = evaluate_videos(
            test_structure, tflite_model_path, args.img_roi, args.num_classes
        )
        no_stable_segments_videos_all_folds[-1][1].extend(additional_no_stable_segments_videos)
        video_accuracy = np.trace(video_conf_matrix) / np.sum(video_conf_matrix) if np.sum(video_conf_matrix) else 0

        # Print evaluation results
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

        # Record fold data
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
            'Row Weighted Scores': [list(ws) for ws in row_weighted_scores],  # Record each row's weighted scores
            'Row Total Weighted Scores': row_total_weighted_scores  # Record each row's total weighted scores
        }
        if len(args.subfolders) == 5:
            fold_data['Cut-off 10^2 CFU/ml'] = accuracy_cutoff_revised

        fold_data_df = pd.DataFrame([fold_data])  # Convert fold data to DataFrame
        results = pd.concat([results, fold_data_df], ignore_index=True)
        
        # Append individual results to summary
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

    # Use cumulative results
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

    # Calculate total row weighted scores across all folds
    combined_row_weighted_scores = np.sum([np.array(scores) for scores in results_summary['Row Weighted Scores']], axis=0)
    combined_row_total_weighted_scores = np.sum([np.array(scores) for scores in results_summary['Row Total Weighted Scores']], axis=0)

    # Print cumulative results
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

    # Create a DataFrame for cumulative results
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

    # Create DataFrames for cumulative confusion matrices
    total_frame_conf_matrix_df = pd.DataFrame(
        total_frame_conf_matrix, columns=[f"Pred_{i}" for i in range(total_frame_conf_matrix.shape[1])]
    )
    total_frame_conf_matrix_df.index = [f"True_{i}" for i in range(total_frame_conf_matrix_df.shape[0])]
    total_video_conf_matrix_df = pd.DataFrame(
        total_video_conf_matrix, columns=[f"Pred_{i}" for i in range(total_video_conf_matrix.shape[1])]
    )
    total_video_conf_matrix_df.index = [f"True_{i}" for i in range(total_video_conf_matrix_df.shape[0])]

    # Save to Excel 
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

        # Save combined row weighted scores
        combined_weighted_scores_df = pd.DataFrame(combined_row_weighted_scores, columns=[f"Weight_{i}" for i in range(combined_row_weighted_scores.shape[1])])
        combined_weighted_scores_df.index = [f"Class_{i}" for i in range(combined_row_weighted_scores.shape[0])]
        combined_weighted_scores_df.to_excel(writer, sheet_name='Row Weighted Scores', index=True)

        combined_total_weighted_scores_df = pd.DataFrame(combined_row_total_weighted_scores, columns=['Total Weighted Score'])
        combined_total_weighted_scores_df.index = [f"Class_{i}" for i in range(combined_row_total_weighted_scores.shape[0])]
        combined_total_weighted_scores_df.to_excel(writer, sheet_name='Row Total Weighted Scores', index=True)

        # Save list of videos with no stable segments
        no_stable_segments_videos_df = pd.DataFrame(columns=['Fold', 'Date Folder', 'Subfolder', 'Video Name', 'Reason'])
        for fold_index, videos in no_stable_segments_videos_all_folds:
            for date_folder, subfolder, video_name_folder, reason in videos:
                video_data = {
                    'Fold': fold_index,
                    'Date Folder': date_folder,
                    'Subfolder': subfolder,
                    'Video Name': video_name_folder,
                    'Reason': reason
                }
                no_stable_segments_videos_df = no_stable_segments_videos_df.append(video_data, ignore_index=True)
        no_stable_segments_videos_df.to_excel(writer, sheet_name='No Stable Segments Videos', index=False)

    print(f"Results and averages have been saved to {excel_path}")
    print("\nVideos with No Stable Segments in Each Fold:")
    for fold_index, videos in no_stable_segments_videos_all_folds:
        if videos:
            print(f"Fold {fold_index}:")
            for date_folder, subfolder, video_name_folder, reason in videos:
                print(f"  - Date Folder: {date_folder}, Subfolder: {subfolder}, Video: {video_name_folder}, Reason: {reason}")

                
def parse_arguments():
    parser = argparse.ArgumentParser(description="Script to train and evaluate a convolutional neural network on image data.")
    parser.add_argument('--base_path', type=str, default='/home/jijang/projects/Bacteria/dataset/case_test/case1', help='Base directory for the dataset.')
    parser.add_argument('--models_dir', type=str, default='/home/jijang/projects/Bacteria/models/case_test/240802_case1_tf', help='Directory where models are saved.')
    parser.add_argument('--excel_dir', type=str, default='/home/jijang/projects/Bacteria/excel/case_test/240820_case1_tf',  help='Directory where excels are saved.')
    parser.add_argument('--subfolders', type=str, nargs='+', default=['0', '1', '2', '3'], help='Subfolders to include as classes.')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes to predict.')
    parser.add_argument('--img_frame', type=int, default=900, help='Frame size of the images.')
    parser.add_argument('--img_roi', type=int, default=96, help='Region of interest size for each image.')
    parser.add_argument('--stability_threshold', type=int, default=350, help='Threshold for stability in video segmentation.')
    parser.add_argument('--buffer_size', type=int, default=0, help='Buffer size around detected peaks in stability analysis.') 
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
