import os
# GPU 설정 (train과 다르게 GPU 디바이스가 4로 지정됨)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "4"         # GPU 4번만 사용
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'         # INFO 메시지 및 경고 숨김

import imageio
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model 
from scipy.signal import find_peaks
import pandas as pd
from tensorflow.keras import backend as K
from skimage import exposure 

# FLOPs/MACs 분석을 위해 필요한 모듈 (TF2.x 환경에서 compat 모듈 사용)
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

###############################################################################
# (1) 재현성(Seed 고정)
###############################################################################
def set_seed(seed):
    """
    실험의 재현성을 위해 numpy와 TensorFlow random seed를 고정합니다.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(42)

###############################################################################
# (2) 모델 FLOPs 계산 함수
###############################################################################
def get_flops(model, batch_size=1):
    """
    주어진 Keras 모델에 대한 총 FLOPs(부동소수점 연산 수)를 계산합니다.
    
    - model: Keras 모델
    - batch_size: 배치 크기 (기본값=1)
    
    동작 원리:
      1) 모델에 입력 텐서를 주어 concrete function을 생성
      2) convert_variables_to_constants_v2 함수로 그래프를 고정(frozen)
      3) tf.compat.v1.profiler를 사용해 FLOPs를 추정
    """
    # 모델을 TF function으로 래핑
    concrete = tf.function(lambda x: model(x))
    # 모델 입력 스펙을 정의
    input_shape = [batch_size] + list(model.input.shape[1:])
    concrete_func = concrete.get_concrete_function(tf.TensorSpec(input_shape, model.input.dtype))

    # 변수(graph) 고정
    frozen_func = convert_variables_to_constants_v2(concrete_func)
    graph_def = frozen_func.graph.as_graph_def()

    # 프로파일링
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name="")
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(
            graph=graph, 
            run_meta=run_meta, 
            cmd="op", 
            options=opts
        )
    return flops.total_float_ops

###############################################################################
# (3) 이미지 로드( TIFF / YUV ) 함수
###############################################################################
def load_tiff_image(file_path, apply_equalization):
    """
    TIFF 이미지를 불러와 float32 범위(0~1)로 정규화 후,
    필요 시 히스토그램 평활화를 적용해 반환합니다.
    """
    image = imageio.imread(file_path)                       # TIFF 파일 로딩
    image = image.astype(np.float32) / 65535.0             # 16-bit TIFF 가정 (65535로 나눠 0~1)
    if apply_equalization:
        image = exposure.equalize_hist(image)              # 히스토그램 평활화
    return image

def load_yuv_image(file_path, apply_equalization=False):
    """
    YUV(8bit, 128x128) 형식의 파일을 로드한 뒤 0~1 범위로 스케일링.
    필요 시 히스토그램 평활화를 적용합니다.
    """
    try:
        file_size = os.path.getsize(file_path)
        required_size = 128 * 128  # 8bit Gray: 128x128
        
        # 파일 크기 검증
        if file_size < required_size:
            raise ValueError(f"File '{file_path}' size ({file_size} bytes) is smaller than required size ({required_size} bytes).")

        with open(file_path, 'rb') as file:
            raw_data = file.read(required_size)
        if len(raw_data) != required_size:
            raise ValueError(
                f"Read data size ({len(raw_data)} bytes) from file '{file_path}' "
                f"does not match required size ({required_size} bytes)."
            )

        # YUV -> (128,128) float32
        image = np.frombuffer(raw_data, dtype=np.uint8).reshape((128, 128))
        image = image.astype(np.float32) / 255.0

        # 히스토그램 평활화 옵션
        if apply_equalization:
            image = exposure.equalize_hist(image)
        return image
    except Exception as e:
        print(f"Error loading YUV image from file '{file_path}': {e}")
        raise

###############################################################################
# (3.1) 다양한 이미지 파일 리스트를 받아 일괄 처리
###############################################################################
def get_images(file_paths, apply_equalization):
    """
    주어진 file_paths 목록에 있는 이미지 파일(TIFF/YUV)을 차례로 로드하여
    ndarray로 묶어 반환합니다.
    """
    images = []
    for file_path in file_paths:
        if file_path.endswith('.tiff'):
            image = load_tiff_image(file_path, apply_equalization)
        else:
            image = load_yuv_image(file_path, apply_equalization)
        images.append(image)
    return np.array(images)

###############################################################################
# (4) train과 같은 방식으로 테스트 구조를 만들어, 안정 세그먼트를 기반으로 X, y 데이터 생성
###############################################################################
def process_images(structure, img_roi, num_classes, apply_equalization):
    """
    생성된 structure( 안정 세그먼트 단위 ) 정보를 이용해
    1) 이미지를 불러와 차분(X_data)
    2) 라벨(y_data)을 생성
    3) ROI crop, one-hot 변환

    - structure: { date_folder: { subfolder: { video_folder: {segment_id: [file_paths]}}}}
    - img_roi: ROI 크기(예: 96)
    - num_classes: 클래스 개수
    - apply_equalization: 히스토그램 평활화 여부
    """
    X_data = []
    y_data = []

    # 구조 순회
    for date_folder, subfolders in structure.items():
        for subfolder, video_folders in subfolders.items():
            for video_folder, segments in video_folders.items():
                for segment_id, file_paths in segments.items():
                    # 세그먼트 내 프레임이 1개 이하이면 차분 불가
                    if len(file_paths) <= 1:
                        continue

                    # TIFF/YUV 로딩
                    X = get_images(file_paths, apply_equalization)

                    # 라벨: 세그먼트 내 (프레임개수 - 1)만큼
                    for _ in range(len(file_paths) - 1):
                        y_data.append(int(subfolder))

                    # 순차 차분 (X - X.roll(-1))
                    X_processed = abs(X - np.roll(X, -1, axis=0))
                    # 마지막 차분은 유효하지 않으므로 제거
                    X_processed = np.delete(X_processed, -1, axis=0)
                    X_data.append(X_processed)

    if not X_data:
        raise ValueError("No data found. Please check your dataset structure.")

    # 리스트를 하나의 (N, H, W) 배열로 변환
    X_data = np.vstack(X_data).astype(np.float32)
    # Gray scale 이미지를 (N, H, W, 1)로 reshape
    X_data = np.expand_dims(X_data, axis=-1)

    # 라벨을 one-hot
    y_data = to_categorical(y_data, num_classes)

    # ROI crop
    X_data = X_data[:, :img_roi, :img_roi, :]

    return X_data, y_data

###############################################################################
# (5) 큰 움직임(피크) 주변을 제외한 안정 프레임 인덱스를 찾는 함수
###############################################################################
def find_all_stable_segments(summed_diffs, peaks, buffer_size):
    """
    - summed_diffs: 프레임별 차분 합 (길이= total_frames-1)
    - peaks: 움직임이 큰 지점(인덱스들)
    - buffer_size: 피크 주변을 제외할 범위
    
    1) 피크 ± buffer_size를 excluded로 등록
    2) 제외되지 않은 프레임 인덱스를 stable로 간주
    """
    total_frames = len(summed_diffs)
    excluded_indices = set()

    for peak in peaks:
        for i in range(max(0, peak - buffer_size), min(total_frames, peak + buffer_size + 1)):
            excluded_indices.add(i)

    # 제외 목록에 없는 인덱스만 반환
    valid_frames = [i for i in range(total_frames) if i not in excluded_indices]
    return valid_frames

###############################################################################
# (6) 안정 세그먼트를 구조화하여 반환
###############################################################################
def get_temporary_structure_with_stable_segments(base_path, date_folders, subfolders,
                                                stability_threshold, buffer_size, equalize):
    """
    - train과 동일하게: date_folders, subfolders를 순회
    - 각 비디오 폴더 내 TIFF/YUV 파일들로부터 안정 세그먼트 인덱스를 구해 structure에 담음
    - 안정 프레임이 없는(no_stable_segments) 비디오는 리스트로 수집
    """
    structure = {}
    no_stable_segments_videos = []  # 안정적인 세그먼트가 없는 비디오 기록

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
                            # TIFF/YUV 파일 수집 후 정렬
                            video_files = [
                                f for f in os.listdir(video_folder_path)
                                if f.endswith('.yuv') or f.endswith('.tiff')
                            ]
                            video_files_sorted = sorted(video_files)

                            # 안정 프레임 인덱스 찾기
                            stable_segments = process_video_files(
                                video_folder_path,
                                video_files_sorted,
                                stability_threshold,
                                buffer_size,
                                equalize
                            )
                            # 안정 프레임이 있으면 세그먼트 구조화
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
                                # 마지막 구간 마무리
                                end = stable_segments[-1] + 1
                                segment_files = [
                                    os.path.join(video_folder_path, video_files_sorted[j])
                                    for j in range(start, end)
                                ]
                                structure[date_folder][subfolder][video_name_folder][f"segment_{segment_id}"] = segment_files
                            else:
                                # 안정 프레임이 없다면 리스트에 기록
                                no_stable_segments_videos.append((date_folder, subfolder, video_name_folder, "No stable segments"))
                                print(f"No stable segments found in {video_name_folder}")

    return structure, no_stable_segments_videos

###############################################################################
# (7) 각 비디오 내 TIFF/YUV 파일을 순서대로 로드 -> 차분 합 -> 피크 찾기 -> 안정 구간
###############################################################################
def process_video_files(folder_path, file_names, stability_threshold, buffer_size, apply_equalization):
    """
    각 영상 폴더 안에 있는 파일 리스트(file_names)를 순서대로 불러
    1) 인접 프레임 차분 -> summed_diffs
    2) find_peaks로 움직임 큰 지점(피크) 찾기
    3) find_all_stable_segments로 안정 구간 인덱스 반환
    """
    if not file_names:
        print("The folder is empty. No files to process.")
        return []

    # 확장자 판단(TIFF / YUV)
    if file_names[0].endswith('.yuv'):
        images = load_yuv_images(folder_path, file_names, apply_equalization)
    elif file_names[0].endswith('.tiff'):
        images = load_tiff_images(folder_path, file_names, apply_equalization)
    else:
        raise ValueError("Unsupported file format")

    # 차분
    image_diffs = np.abs(images[:-1] - images[1:])
    summed_diffs = image_diffs.sum(axis=(1, 2))

    # 임계값(stability_threshold) 이상인 지점을 피크
    peaks, _ = find_peaks(summed_diffs, height=stability_threshold)
    # 피크 주변 제외 -> 안정 구간
    stable_segments = find_all_stable_segments(summed_diffs, peaks, buffer_size)
    return stable_segments

###############################################################################
# (7.1) TIFF/YUV를 한 번에 로드하기 위한 함수
###############################################################################
def load_tiff_images(folder_path, file_names, equalize):
    """
    folder_path 내 file_names 목록에 있는 TIFF를 차례로 load_tiff_image로 불러와
    np.array로 반환
    """
    images = []
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        image = load_tiff_image(file_path, equalize)
        images.append(image)
    return np.array(images)

def load_yuv_images(folder_path, file_names, apply_equalization, image_size=(128, 128)):
    """
    folder_path 내 file_names 목록에 있는 YUV 파일을 load_yuv_image로 불러와 
    np.array로 반환
    """
    images = []
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        image = load_yuv_image(file_path, apply_equalization)
        images.append(image)
    return np.array(images)

###############################################################################
# (8) 모델 평가 함수: (프레임 단위) 혼동 행렬 및 가중치 점수 계산
###############################################################################
def evaluate_model(model, X_test, y_test, num_classes):
    """
    주어진 Keras model에 대해 X_test, y_test(원-핫 라벨)로 추론 후,
    혼동 행렬, 정확도, 다양한 가중치 점수(row_weighted_scores 등)를 계산합니다.
    """
    # 예측
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)

    # 혼동 행렬
    conf_matrix = confusion_matrix(y_test, y_pred, labels=range(num_classes))

    # 분류 가중치 설정 (예: 10^i)
    weights = np.array([10 ** i for i in range(num_classes)], dtype=np.float32)
    # weights[0] = 0 => 0번 클래스에 대해서는 가중치 0
    weights[0] = 0

    # 정확도
    accuracy = np.mean(y_pred == y_test)

    row_weighted_scores = []
    row_total_weighted_scores = []

    # 행(진짜 클래스)별로 가중치 계산
    for row in conf_matrix:
        total_samples = np.sum(row)
        if total_samples == 0:
            row_weighted_scores.append([0] * len(weights))
            row_total_weighted_scores.append(0)
            continue

        # row에서 가장 많이 예측된 클래스(Top1) / 그 값
        top1_index = np.argmax(row)
        top1_value = row[top1_index]

        # 두 번째로 많이 예측된 클래스(Top2)
        row_excluding_top1 = np.copy(row)
        row_excluding_top1[top1_index] = 0
        top2_index = np.argmax(row_excluding_top1)
        top2_value = row_excluding_top1[top2_index]

        # Top1 가중치 점수
        top1_scores = top1_value * weights[top1_index] / total_samples

        # Top2가 인접 클래스면 추가 가중치 반영
        if abs(top2_index - top1_index) == 1:
            top2_scores = top2_value * weights[top2_index] / total_samples
            total_weighted_score = top1_scores + top2_scores
        else:
            total_weighted_score = top1_scores

        row_weighted_scores.append(
            [row[i] * weights[i] / total_samples for i in range(len(row))]
        )
        row_total_weighted_scores.append(total_weighted_score)

    return accuracy, conf_matrix, row_weighted_scores, row_total_weighted_scores

###############################################################################
# (8.1) 프레임 전처리 함수 ( 단일 비디오에서 테스트 용 )
###############################################################################
def preprocess_data(file_paths, img_roi, apply_equalization):
    """
    file_paths에 있는 프레임들을 로딩 -> 인접 프레임 차분 -> ROI crop
    -> (N, roi, roi, 1) shape의 np.array로 반환
    """
    X = get_images(file_paths, apply_equalization)
    X = np.array(X)

    # 인접 프레임 차분
    X_tests = abs(X - np.roll(X, -1, axis=0))
    X_tests = np.delete(X_tests, -1, axis=0)

    X_test = np.expand_dims(X_tests, axis=-1).astype(np.float32)
    X_test = X_test[:, :img_roi, :img_roi, :]
    return X_test

###############################################################################
# (8.2) TFLite 추론 함수 (Optional: TensorRT delegate)
###############################################################################
def model_inference(X_test, tflite_model_path, use_trt=False):
    """
    - X_test: (N, roi, roi, 1)
    - tflite_model_path: 변환된 TFLite 모델 경로
    - use_trt: TensorRT delegate 사용 여부
    """
    if use_trt:
        # TensorRT delegate를 로드
        try:
            try:
                from tensorflow.lite.experimental import load_delegate
            except ImportError:
                from tensorflow.lite.python.interpreter import load_delegate
            
            trt_delegate = load_delegate('libnvinfer_plugin.so')
            interpreter = tf.lite.Interpreter(
                model_path=tflite_model_path, 
                experimental_delegates=[trt_delegate]
            )
            print("Using TensorRT delegate for inference.")
        except Exception as e:
            print("Error loading TensorRT delegate:", e)
            print("Falling back to default interpreter.")
            interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    else:
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)

    # Interpreter 초기화
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    output_data = []
    # 배치가 아닌 1개씩 추론
    for i in range(X_test.shape[0]):
        single_input = np.expand_dims(X_test[i], axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], single_input)
        interpreter.invoke()
        single_output = interpreter.get_tensor(output_details[0]['index'])
        # argmax로 라벨 결정
        single_output = np.argmax(single_output, axis=1)
        output_data.append(single_output)

    output_data = np.array(output_data)
    return output_data

###############################################################################
# (8.3) np.array에서 최빈값(가장 많이 등장한 값) 찾기
###############################################################################
def find_mode(lst):
    """
    1D array로 flatten한 뒤 Counter로 최빈값(가장 많이 등장하는 값)을 반환
    """
    lst = lst.reshape(-1)
    from collections import Counter
    counter = Counter(lst)
    most_common = counter.most_common(1)
    return most_common[0][0]

###############################################################################
# (9) 비디오 단위 예측 -> 혼동행렬 계산
###############################################################################
def evaluate_videos(structure, tflite_model_path, img_roi, num_classes, apply_equalization, use_trt=False):
    """
    - structure: 안정 세그먼트별 파일 경로
    - tflite_model_path: TFLite 모델 경로
    - img_roi: ROI 크기
    - num_classes: 클래스 개수
    - apply_equalization: 히스토그램 평활화 여부
    - use_trt: TensorRT delegate 사용 여부

    각 (date_folder, subfolder, video_folder)에 대해:
      1) segment_id마다 프레임 차분 -> tflite 추론 -> 최빈값
      2) 세그먼트별 라벨 리스트 -> 다시 최빈값 -> 최종 비디오 라벨
      3) 실제 subfolder와 매칭 -> 혼동행렬 누적
    """
    confusion_matrix_result = np.zeros((num_classes, num_classes), dtype=int)
    video_results = {}    # {video_folder: [라벨1, 라벨2, ...]}
    actual_classes = {}   # {video_folder: subfolder(실제)}
    insufficient_frame_videos = set()

    for date_folder, subfolders in structure.items():
        for subfolder, video_folders in subfolders.items():
            for video_folder, segments in video_folders.items():
                insufficient_frames_count = 0
                for segment_id, file_paths in segments.items():
                    # 세그먼트 프레임이 충분치 않으면 스킵
                    if (file_paths is None) or (len(file_paths) <= 1):
                        insufficient_frames_count += 1
                        continue
                    images = preprocess_data(file_paths, img_roi, apply_equalization)
                    # TFLite 추론 후 모드
                    result = model_inference(images, tflite_model_path, use_trt)
                    most_common_result = find_mode(result)

                    if video_folder not in video_results:
                        video_results[video_folder] = []
                        actual_classes[video_folder] = subfolder
                    video_results[video_folder].append(most_common_result)

                # 모든 세그먼트가 프레임 부족이면 insufficient_frames
                if insufficient_frames_count == len(segments):
                    insufficient_frame_videos.add((date_folder, subfolder, video_folder, "Insufficient frames"))

    # 비디오 단위 최종 라벨
    for video_folder, results in video_results.items():
        final_class = find_mode(np.array(results))
        subfolder = actual_classes[video_folder]
        # 혼동행렬 누적
        confusion_matrix_result[int(subfolder), final_class] += 1

    return confusion_matrix_result, list(insufficient_frame_videos)

###############################################################################
# (10) 특정 컷오프(10^2 CFU/ml)에 대한 Accuracy 계산
###############################################################################
def calculate_cutoff_accuracy(conf_matrix):
    """
    클래스 0~1을 '하위 범위',
    클래스 2~ 이상을 '상위 범위'라 가정하고,
    하위 범위(0~1)끼리는 TP, 상위 범위(2~...)끼리는 TP로 계산,
    총 TP / 전체 샘플 로서 accuracy 산출.
    """
    # 상위 범위: 클래스 2 이상
    tp_top_left = np.sum(conf_matrix[0:2, 0:2])  # true:0-1, pred:0-1
    tp_bottom_right = np.sum(conf_matrix[2:, 2:])  # true:2+, pred:2+
    total_true_positives = tp_top_left + tp_bottom_right

    total_samples = np.sum(conf_matrix)
    accuracy = total_true_positives / total_samples if total_samples else 0
    return accuracy

###############################################################################
# (11) F1 스코어 계산(클래스별)
###############################################################################
def calculate_f1_scores(conf_matrix):
    """
    각 클래스(i)에 대해:
      precision = tp / (tp + fp)
      recall    = tp / (tp + fn)
      f1        = 2 * (precision * recall) / (precision + recall)
    """
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

###############################################################################
# (12) 메인 실행: K-Fold 평가 진행 ( 날짜 폴더 = Fold )
###############################################################################
def main(args):
    """
    1) date_folders를 fold로 가정, 테스트 폴더를 교차로 지정
    2) 안정 프레임 구성 -> X_test, y_test
    3) 모델 로드 후 프레임/비디오 수준 평가
    4) 혼동 행렬, F1, 등등을 Excel에 저장
    """
    # 엑셀 파일 초기화
    os.makedirs(args.excel_dir, exist_ok=True)
    excel_path = os.path.join(args.excel_dir, "test_results.xlsx")
    df_dummy = pd.DataFrame()
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_dummy.to_excel(writer)

    # 날짜 폴더 (fold) 탐색
    date_folders = sorted([d for d in os.listdir(args.base_path) if os.path.isdir(os.path.join(args.base_path, d))])
    print(f"Total date folders found: {date_folders}")

    if not date_folders:
        raise ValueError("No date folders found in the base path. Please check the dataset directory.")

    # 결과 저장할 DataFrame
    results = pd.DataFrame(columns=[
        'Fold', 'Train Folders', 'Test Folder', 'Frame Confusion Matrix', 
        'Frame Accuracy', 'Video Confusion Matrix', 'Video Accuracy', 
        'Cut-off 10^2 CFU/ml', 'Frame F1 Score', 'Video F1 Score',
        'Row Weighted Scores', 'Row Total Weighted Scores'
    ])

    # 요약 정보 저장용 딕셔너리
    results_summary = {
        'Frame Accuracy': [],
        'Video Accuracy': [],
        'Cut-off 10^2 CFU/ml': [],
        'Frame F1 Score': [],
        'Video F1 Score': [],
        'Row Weighted Scores': [],
        'Row Total Weighted Scores': []
    }

    # 누적 혼동행렬(프레임/비디오)
    cumulative_frame_conf_matrix = np.zeros((args.num_classes, args.num_classes))
    cumulative_video_conf_matrix = np.zeros((args.num_classes, args.num_classes))

    # 안정 구간 없는 비디오 기록
    no_stable_segments_videos_all_folds = []

    # K-Fold Cross Validation: date_folders 개수만큼 fold
    for fold_index, test_folder in enumerate(date_folders):
        print(f"Starting Fold {fold_index + 1} - Test Folder: {test_folder}")

        # 테스트 폴더: test_folder
        val_folder = [test_folder]
        # 나머지 폴더: train_folders
        train_folders = [folder for folder in date_folders if folder != test_folder]

        print(f"Train Folders: {train_folders}")
        print(f"Validation Folder: {val_folder}")

        # 테스트 폴더에 대한 구조 생성 (안정 프레임)
        val_structure, no_stable_segments_videos = get_temporary_structure_with_stable_segments(
            args.base_path, val_folder, args.subfolders, 
            args.stability_threshold, args.buffer_size, args.equalize
        )
        no_stable_segments_videos_all_folds.append((fold_index + 1, no_stable_segments_videos))

        # X_test, y_test 생성
        try:
            X_test, y_test = process_images(val_structure, args.img_roi, args.num_classes, args.equalize)
        except ValueError as e:
            print(f"Error processing images for Fold {fold_index + 1}: {e}")
            continue

        if len(X_test) == 0:
            print(f"No data available for Fold {fold_index + 1}. Skipping this fold.")
            continue

        print(f"Fold {fold_index + 1}: Test data shape: {X_test.shape}, Test labels shape: {y_test.shape}")

        # 모델 경로 (train 때와 동일한 naming)
        model_path = os.path.join(args.models_dir, f"Fold_{fold_index + 1}_model.h5")
        if not os.path.exists(model_path):
            print(f"Model file {model_path} does not exist. Skipping Fold {fold_index + 1}.")
            continue

        # Keras 모델 로드
        model = load_model(model_path)

        # FLOPs/MACs 분석
        try:
            total_flops = get_flops(model, batch_size=1)
            total_macs = total_flops / 2
            print(f"Fold {fold_index + 1}: Model FLOPs: {total_flops}")
            print(f"Fold {fold_index + 1}: Model MACs (approx.): {total_macs}")
        except Exception as e:
            print(f"Fold {fold_index + 1}: Error computing FLOPs/MACs: {e}")

        # 학습 가중치(파라미터) 수
        trainable_params = np.sum([K.count_params(w) for w in model.trainable_weights])
        print(f"Fold {fold_index + 1}: Number of trainable parameters: {trainable_params}")

        # 모델 파일 크기
        model_size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"Fold {fold_index + 1}: Model size: {model_size:.2f} MB")

        # (A) 프레임 수준 평가
        frame_accuracy, frame_conf_matrix, row_weighted_scores, row_total_weighted_scores = evaluate_model(
            model, X_test, y_test, args.num_classes
        )
        # 폴드별 혼동행렬 누적
        cumulative_frame_conf_matrix += frame_conf_matrix

        # (B) TFLite 변환
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS, 
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            tflite_model = converter.convert()
            tflite_model_path = os.path.join(args.models_dir, f"Fold_{fold_index + 1}_HTCNN.tflite")
            with open(tflite_model_path, 'wb') as f:
                f.write(tflite_model)
            print(f"TFLite 모델이 성공적으로 저장되었습니다: {tflite_model_path}")
        except Exception as e:
            print(f"TFLite 변환 중 오류가 발생했습니다: {e}")

        # (C) 비디오 단위 평가 (TFLite 추론)
        video_conf_matrix, additional_no_stable_segments_videos = evaluate_videos(
            val_structure, 
            tflite_model_path, 
            args.img_roi, 
            args.num_classes, 
            args.equalize, 
            use_trt=args.use_trt
        )
        # "No stable segments"에 추가 항목이 있다면 추가
        no_stable_segments_videos_all_folds[-1][1].extend(additional_no_stable_segments_videos)

        # 비디오 혼동행렬 누적
        cumulative_video_conf_matrix += video_conf_matrix
        video_accuracy = np.trace(video_conf_matrix) / np.sum(video_conf_matrix) if np.sum(video_conf_matrix) else 0

        # 출력
        print(f"Evaluating Fold {fold_index + 1}: test {test_folder}")
        print(f"Frame Confusion Matrix for Fold {fold_index + 1}:\n{frame_conf_matrix}")
        print(f"Frame Accuracy for Fold {fold_index + 1}: {frame_accuracy * 100:.2f}%\n")
        print(f"Row Weighted Scores for Fold {fold_index + 1}: {row_weighted_scores}")
        print(f"Row Total Weighted Scores for Fold {fold_index + 1}: {row_total_weighted_scores}\n")

        print(f"Video Confusion Matrix for Fold {fold_index + 1}:\n{video_conf_matrix}")
        print(f"Video Accuracy for Fold {fold_index + 1}: {video_accuracy * 100:.2f}%\n")

        # F1 스코어 계산(프레임/비디오)
        frame_f1_scores = calculate_f1_scores(frame_conf_matrix)
        video_f1_scores = calculate_f1_scores(video_conf_matrix)

        print(f"Frame F1 Scores for Fold {fold_index + 1}: {frame_f1_scores}") 
        print(f"Video F1 Scores for Fold {fold_index + 1}: {video_f1_scores}") 

        # 5 클래스일 때 0~1 vs 2~4 구분하여 Cut-off 계산
        if len(args.subfolders) == 5:
            accuracy_cutoff_revised = calculate_cutoff_accuracy(video_conf_matrix)
            print(f'Cut-off 10^2 CFU/ml Accuracy for Fold {fold_index + 1}: {accuracy_cutoff_revised * 100:.2f}%\n')
        else:
            accuracy_cutoff_revised = None

        test_folder_name = test_folder
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
            'Row Total Weighted Scores': row_total_weighted_scores
        }

        # fold 결과를 결과 DF에 추가
        fold_data_df = pd.DataFrame([fold_data])
        results = pd.concat([results, fold_data_df], ignore_index=True)
        
        # 요약
        results_summary['Frame Accuracy'].append(frame_accuracy)
        results_summary['Video Accuracy'].append(video_accuracy)
        results_summary['Cut-off 10^2 CFU/ml'].append(accuracy_cutoff_revised if accuracy_cutoff_revised is not None else 0)
        results_summary['Frame F1 Score'].append(frame_f1_scores)
        results_summary['Video F1 Score'].append(video_f1_scores)
        results_summary['Row Weighted Scores'].append(row_weighted_scores)
        results_summary['Row Total Weighted Scores'].append(row_total_weighted_scores)

    # 모든 fold 종료 후 전체 평균 계산
    total_frame_accuracy = np.mean(results_summary['Frame Accuracy']) if results_summary['Frame Accuracy'] else 0
    total_video_accuracy = np.mean(results_summary['Video Accuracy']) if results_summary['Video Accuracy'] else 0

    if (len(args.subfolders) == 5 and results_summary['Cut-off 10^2 CFU/ml']):
        total_cutoff_accuracy = np.mean(results_summary['Cut-off 10^2 CFU/ml'])
    else:
        total_cutoff_accuracy = None

    # 누적 혼동행렬을 이용한 F1 스코어
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

    # 최종 요약 DF
    total_results = pd.DataFrame({
        'Metric': [
            'Frame Accuracy',
            'Video Accuracy',
            'Cut-off 10^2 CFU/ml Accuracy', 
            'Average Frame F1 Score',
            'Average Video F1 Score'
        ],
        'Value': [
            f"{total_frame_accuracy * 100:.2f}%",
            f"{total_video_accuracy * 100:.2f}%",
            f"{total_cutoff_accuracy * 100:.2f}%" if total_cutoff_accuracy is not None else "N/A",
            f"{total_frame_f1_score}",
            f"{total_video_f1_score}"
        ]
    })

    # 혼동행렬 DF 변환
    total_frame_conf_matrix_df = pd.DataFrame(
        cumulative_frame_conf_matrix,
        columns=[f"Pred_{i}" for i in range(cumulative_frame_conf_matrix.shape[1])]
    )
    total_frame_conf_matrix_df.index = [f"True_{i}" for i in range(total_frame_conf_matrix_df.shape[0])]

    total_video_conf_matrix_df = pd.DataFrame(
        cumulative_video_conf_matrix,
        columns=[f"Pred_{i}" for i in range(cumulative_video_conf_matrix.shape[1])]
    )
    total_video_conf_matrix_df.index = [f"True_{i}" for i in range(total_video_conf_matrix_df.shape[0])]

    # Excel 저장
    results.to_excel(excel_path, index=False)  # fold별 상세 결과
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

    # 안정 구간 없는 비디오 리스트 기록
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

###############################################################################
# (13) 인자 파싱 함수
###############################################################################
def parse_arguments():
    """
    스크립트 실행 시 인자를 파싱하여 Namespace로 반환합니다.
    """
    parser = argparse.ArgumentParser(
        description="Script to evaluate a CNN on image data with optional TensorRT inference using TFLite."
    )
    parser.add_argument('--base_path', type=str,
                        default='/home/jijang/ssd_data/projects/Bacteria/dataset/20231221_DS/Stationary',
                        help='Base directory for the dataset.')
    parser.add_argument('--models_dir', type=str,
                        default='/home/jijang/ssd_data/projects/Bacteria/models/20231221_DS/Stationary',
                        help='Directory where models are saved.')
    parser.add_argument('--excel_dir', type=str,
                        default='/home/jijang/ssd_data/projects/Bacteria/excel/20231221_DS/Stationary',
                        help='Directory where excels are saved.')
    parser.add_argument('--subfolders', type=str, nargs='+',
                        default=['0', '1', '2', '3'],
                        help='Subfolders to include as classes.')
    parser.add_argument('--num_classes', type=int, default=4,
                        help='Number of classes to predict.')
    parser.add_argument('--img_frame', type=int, default=300,
                        help='Frame size of the images.')
    parser.add_argument('--img_roi', type=int, default=96,
                        help='Region of interest size for each image.')
    parser.add_argument('--stability_threshold', type=int, default=350,
                        help='Threshold for stability in video segmentation.')
    parser.add_argument('--buffer_size', type=int, default=2,
                        help='Buffer size around detected peaks in stability analysis.')
    parser.add_argument('--equalize', action='store_true',
                        help='Apply histogram equalization to images.')
    parser.add_argument('--use_trt', action='store_true',
                        help='Use TensorRT delegate for TFLite inference')
    return parser.parse_args()

###############################################################################
# (14) 엔트리 포인트
###############################################################################
if __name__ == "__main__":
    args = parse_arguments()
    main(args)
