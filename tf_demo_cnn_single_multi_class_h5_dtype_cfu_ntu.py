import os
# CUDA 환경 설정
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
os.environ['CUDA_VISIBLE_DEVICES'] = "0"         
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'        

import argparse
import tensorflow as tf
from collections import Counter
import pandas as pd
from PIL import Image
from skimage import exposure
from scipy.signal import find_peaks
import numpy as np
from tensorflow.keras import losses

###############################################################################
# (1) 인자 파싱
#     - 스크립트를 CLI로 실행할 때 넘기는 인자들을 정의합니다.
###############################################################################
def parse_arguments():
    """
    명령줄 인자를 정의하고 기본값을 설정한 뒤, 파싱하여 반환합니다.
    
    - --base_path: 데이터가 있는 상위 폴더 경로
    - --subfolders: 서브폴더 이름 목록 (클래스 혹은 구분용 폴더)
    - --models_dir: 사용할 모델(.tflite 또는 .h5)이 있는 폴더 경로
    - --model_type: 모델 형식(tflite / h5)
    - --excel_dir: 결과를 저장할 엑셀 폴더 경로
    - --img_roi: ROI 크기(이미지 크기를 crop해서 사용할 경우)
    - --stability_threshold: 차분값이 임계값 이상이면 불안정(움직임이 큰 지점)
    - --buffer_size: 큰 움직임(피크) 주변으로 제외할 프레임 범위
    - --equalize: 히스토그램 평활화 적용 여부 (True/False)
    - --precision: 모델 추론(양자화) 정밀도 옵션 (fp32, fp16, int8)
    - --a, --b, --c, --d: NTU(탁도) 계산 시 사용할 다항식의 계수
    """
    parser = argparse.ArgumentParser(
        description="Demo (no GT): stable-frames + ensemble for multi-class (5 classes) classification."
    )
    parser.add_argument('--base_path', type=str,
                        default='dataset/2025_bacteria/250305/0305-2',
                        help='Folder path (e.g., 0114-1).')
    parser.add_argument('--subfolders', type=str, nargs='+',
                        default=['0', 'P4', 'P04', 'P34'],
                        help='Names of subfolders (e.g. 0, P1, P2, P3).')
    parser.add_argument('--models_dir', type=str,
                        default='models/case_test/2025_case16_ptq_fp16',
                        help='Directory with model files (.tflite or .h5) for ensemble.')
    parser.add_argument('--model_type', type=str, choices=['tflite', 'h5'], 
                        default='tflite',
                        help='Choose model type: .tflite or .h5')
    parser.add_argument('--excel_dir', type=str,
                        default='excel/2025_bacteria_multi/0305-2',
                        help='Directory to save results Excel.')
    parser.add_argument('--img_roi', type=int,  default=96,
                        help='Crop region size (e.g. 128 or 96).')
    parser.add_argument('--stability_threshold', type=int, default=350,
                        help='Threshold for stable frame detection.')
    parser.add_argument('--buffer_size', type=int, default=2,
                        help='Buffer around large diffs for excluding stable frames.')
    parser.add_argument('--equalize', action='store_true',
                        help='Apply histogram equalization if set.')
    parser.add_argument('--precision', type=str, default='int8', choices=['fp32', 'fp16', 'int8'],
                        help='Precision for model inference.')
    # NTU(탁도) 계산을 위한 다항식 계수들
    parser.add_argument('--a', type=float, default=0, help='Coefficient a for NTU calculation.')
    parser.add_argument('--b', type=float, default=0, help='Coefficient b for NTU calculation.')
    parser.add_argument('--c', type=float, default=1, help='Coefficient c for NTU calculation.')
    parser.add_argument('--d', type=float, default=0, help='Coefficient d for NTU calculation.')
    return parser.parse_args()

###############################################################################
# (2) 이미지 로딩 함수
#     - TIFF 또는 YUV 포맷 파일을 로드하여 0~1 범위의 float32로 변환
#     - 필요 시 히스토그램 평활화(equlize)를 적용
###############################################################################
def load_image(file_path, apply_equalization=False):
    """
    파일 확장자에 따라 TIFF 또는 YUV 이미지를 로드합니다.
    - TIFF: PIL.Image.open으로 열고 0~1 범위로 스케일링
    - YUV: raw 바이너리를 128x128로 reshape하여 0~1 범위로 스케일링
    
    apply_equalization이 True면 skimage.exposure의 equalize_hist를 적용합니다.
    """
    # TIFF 파일 처리
    if file_path.lower().endswith('.tiff'):
        with Image.open(file_path) as img:
            arr = np.array(img).astype(np.float32) / 255.0
    # YUV 파일 처리
    elif file_path.lower().endswith('.yuv'):
        with open(file_path, 'rb') as f:
            raw = f.read(128 * 128)
            arr = np.frombuffer(raw, dtype=np.uint8).reshape((128, 128)).astype(np.float32) / 255.0
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    # 히스토그램 평활화 적용 여부
    if apply_equalization:
        arr = exposure.equalize_hist(arr)
    return arr

###############################################################################
# (2.1) TFLite 모델 추론 헬퍼 함수
#       - TFLite Interpreter에 입력 배치를 주어 추론 결과를 받는다.
###############################################################################
def tflite_predict(interpreter, X):
    """
    tflite 모델(Interpreter)로부터 예측값을 얻는 함수.
    - interpreter: tf.lite.Interpreter 객체
    - X: 입력 데이터 (shape=(N, H, W, C))
    
    TFLite 모델이 특정 batch size(예: 1, 16 등)만 허용하는 경우도 있으므로,
    입력 batch와 모델 batch가 맞지 않으면(예: 1씩 반복) 처리를 해줌.
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    model_batch = input_details[0]['shape'][0]

    # 모델과 입력 배치 크기가 동일하면 한 번에 추론
    if X.shape[0] == model_batch:
        interpreter.set_tensor(input_details[0]['index'], X)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])
    else:
        # 배치 크기가 다르면 1개씩 처리를 반복
        preds = []
        for i in range(X.shape[0]):
            sample = X[i:i+1]  # shape: (1, roi, roi, 1)
            interpreter.set_tensor(input_details[0]['index'], sample)
            interpreter.invoke()
            out = interpreter.get_tensor(output_details[0]['index'])
            preds.append(out[0])
        predictions = np.array(preds)
    return predictions

###############################################################################
# (3) 안정 프레임(stable frames) 찾기
#     - 프레임 간 차분의 합이 stability_threshold 이하인 구간(움직임이 작은 구간)
#     - 차분이 큰 지점은 find_peaks로 찾아 buffer_size만큼 제외
###############################################################################
def find_stable_frames(images, stability_threshold, buffer_size):
    """
    images: shape=(F, H, W) (F개의 프레임)
    stability_threshold: 차분 합이 넘어가는 지점을 피크로 봄
    buffer_size: 피크 주변을 제외할 범위
    
    returns: 안정된 프레임(차분이 작았던 구간)의 인덱스 리스트
    """
    # 프레임이 2개 미만이면 차분 불가
    if images.shape[0] < 2:
        return []

    # 인접 프레임 차분 (F-1, H, W)
    diffs = np.abs(images[:-1] - images[1:])
    # 차분 결과를 각 프레임 간 sum
    summed = diffs.sum(axis=(1, 2))
    # scipy.signal.find_peaks를 통해 움직임이 큰 지점(피크) 찾음
    peaks, _ = find_peaks(summed, height=stability_threshold)

    # 피크 주변(± buffer_size)을 제외 대상에 추가
    excluded = set()
    for peak in peaks:
        start = max(0, peak - buffer_size)
        end = min(len(summed), peak + buffer_size + 1)
        for i in range(start, end):
            excluded.add(i)

    # 제외되지 않은(움직임이 작은) 인덱스들만 리턴
    stable = [i for i in range(len(summed)) if i not in excluded]
    return stable

###############################################################################
# (3.1) NTU(탁도) 계산 관련 함수: 프레임 자기상관 & 다항식 적용
###############################################################################
def autocor(images, n=1):
    """
    images: shape=(F, H, W)
    n: 시차(lag)
    
    각 프레임 i와 i+n에 대해, ravel() 후 corrcoef를 구해 상관계수를 계산함.
    """
    num_images = images.shape[0]
    autocor_values = []
    for i in range(num_images - n):
        correlation = np.corrcoef(images[i].ravel(), images[i + n].ravel())[0, 1]
        autocor_values.append(correlation)
    return np.array(autocor_values)

def calculate_turbidity(acor, a, b, c, d):
    """
    NTU를 계산하기 위한 다항식:
      NTU = a * (acor)^3 + b * (acor)^2 + c * (acor) + d
    
    acor: 최대 자기상관 값
    a,b,c,d: 사용자 정의 계수
    """
    turbidity = a * acor**3 + b * acor**2 + c * acor + d
    return turbidity

###############################################################################
# (4) 폴더 구조 분석 + 영상 로딩
#     - base_path/subfolders 내의 각 video 폴더를 찾고,
#       프레임 단위로 로드 후 안정구간 세그먼트를 구분하여 structure로 저장
###############################################################################
def build_structure(base_path, subfolders, stability_threshold, buffer_size, apply_equalization):
    """
    구조:
    - structure[subfolder][video_dir] = { segment_0: [...], segment_1: [...], ... }
      => 세그먼트별 안정 프레임 인덱스
    - video_frames_dict[(subfolder, video_dir)] = 실제 프레임 numpy 배열
    
    all_videos_list는 (subfolder, vdir) 리스트로 리턴
    """
    structure = {}
    video_frames_dict = {}
    all_videos_list = []
    
    for subfolder in subfolders:
        folder_path = os.path.join(base_path, subfolder)
        if not os.path.isdir(folder_path):
            # 서브폴더가 존재하지 않는 경우 경고
            print(f"[WARN] Subfolder not found: {folder_path}")
            continue

        # 구조 딕셔너리에 서브폴더 키 생성
        structure[subfolder] = {}
        video_dirs = os.listdir(folder_path)

        for vdir in video_dirs:
            video_path = os.path.join(folder_path, vdir)
            if not os.path.isdir(video_path):
                # 파일이거나 유효한 폴더가 아니라면 스킵
                continue

            # (subfolder, vdir)로 식별
            all_videos_list.append((subfolder, vdir))

            # TIFF 또는 YUV 형식의 파일만 추출
            frame_files = sorted([
                f for f in os.listdir(video_path)
                if f.lower().endswith('.tiff') or f.lower().endswith('.yuv')
            ])

            # 프레임 파일이 하나도 없는 경우
            if not frame_files:
                structure[subfolder][vdir] = {}
                continue

            # 프레임(이미지) 로딩
            frames = [load_image(os.path.join(video_path, ff), apply_equalization)
                      for ff in frame_files]
            frames = np.array(frames, dtype=np.float32)

            # 안정 프레임 인덱스 찾기
            stable = find_stable_frames(frames, stability_threshold, buffer_size)
            if not stable:
                # 안정 프레임이 없으면 seg_dict를 비워두고 저장
                structure[subfolder][vdir] = {}
                video_frames_dict[(subfolder, vdir)] = frames
                continue

            # 안정 프레임 인덱스 리스트를 연속 구간으로 분할 -> segment 단위
            seg_dict = {}
            seg_id = 0
            start = stable[0]
            for i in range(1, len(stable)):
                # 연속되지 않는 지점에서 구간 나누기
                if stable[i] != stable[i-1] + 1:
                    end = stable[i-1] + 1
                    seg_dict[f"segment_{seg_id}"] = list(range(start, end))
                    seg_id += 1
                    start = stable[i]
            # 마지막 구간
            end = stable[-1] + 1
            seg_dict[f"segment_{seg_id}"] = list(range(start, end))

            # structure와 video_frames_dict에 저장
            structure[subfolder][vdir] = seg_dict
            video_frames_dict[(subfolder, vdir)] = frames

    return structure, video_frames_dict, all_videos_list

###############################################################################
# (5) 앙상블 평가 + 프레임별 예측 결과 수집
#     - 모든 모델에 대해 프레임 차분 이미지로 추론 -> 다수결
#     - 세그먼트 단위도 다수결 -> 최종 비디오 레벨 예측
###############################################################################
def evaluate_video_level_ensemble_no_gt(structure, video_frames_dict, all_videos_list, models, img_roi, precision):
    """
    structure, video_frames_dict, all_videos_list를 이용해 영상을 순회하면서
    - 각 세그먼트에 대해 프레임(차분)별 예측
    - 모델 앙상블(다수결) 수행
    - 세그먼트 단위로 최종 라벨 -> 비디오 단위로도 최종 라벨
    
    return:
      predictions_dict[subfolder][video] = 최종 비디오 예측(라벨 or 'invalid')
      frame_predictions_list: 프레임 단위 예측 기록(딕셔너리 리스트)
    """
    predictions_dict = {}       # (subfolder, video) -> 최종 예측 라벨
    frame_predictions_list = [] # 프레임단위 예측 기록용 리스트
    
    for (subfolder, vname) in all_videos_list:
        # 해당 subfolder에 대한 dict 초기화
        if subfolder not in predictions_dict:
            predictions_dict[subfolder] = {}

        # structure에서 해당 비디오의 세그먼트 정보 확인
        if subfolder not in structure or vname not in structure[subfolder]:
            # 구조에 없으면 invalid
            predictions_dict[subfolder][vname] = 'invalid'
            continue

        seg_dict = structure[subfolder][vname]
        frames = video_frames_dict.get((subfolder, vname), None)
        if frames is None:
            # 로딩된 프레임이 없으면 invalid
            predictions_dict[subfolder][vname] = 'invalid'
            continue

        # 세그먼트별 예측
        segment_labels = []
        for seg_id, idx_list in seg_dict.items():
            # 안정 프레임 구간이 2개 미만이면 차분할 수 없음
            if len(idx_list) < 2:
                continue

            seg_frames = frames[idx_list]  # shape=(k, H, W)
            # 인접 프레임 차분
            diffs = np.abs(seg_frames[:-1] - seg_frames[1:])
            if diffs.shape[0] == 0:
                continue

            # ROI 크기만큼 crop 후 채널 차원 추가 (Gray -> 1 channel)
            diffs = diffs[:, :img_roi, :img_roi]
            X = np.expand_dims(diffs, axis=-1)  # (num_diff, roi, roi, 1)

            # (A) 프레임별 예측
            for diff_idx in range(X.shape[0]):
                sample = X[diff_idx:diff_idx+1]  # (1, roi, roi, 1)
                frame_model_preds = []
                for model in models:
                    # TFLite / Keras 모델 구분
                    if isinstance(model, tf.lite.Interpreter):
                        p = tflite_predict(model, sample)
                    else:
                        p = model.predict(sample, verbose=0)
                    label = int(np.argmax(p, axis=1)[0])
                    frame_model_preds.append(label)
                
                # 앙상블 다수결
                aggregated_frame_label = Counter(frame_model_preds).most_common(1)[0][0]

                # 프레임별 예측 결과를 frame_predictions_list에 기록
                frame_predictions_list.append({
                    'Folder': subfolder,
                    'Video': vname,
                    'Segment': seg_id,
                    'FramePair': f"{idx_list[diff_idx]}-{idx_list[diff_idx+1]}",
                    'Prediction': aggregated_frame_label
                })

            # (B) 세그먼트 단위 예측 (각 모델별로 프레임 다수결 후 다시 앙상블)
            model_preds = []
            for model in models:
                if isinstance(model, tf.lite.Interpreter):
                    p = tflite_predict(model, X)
                else:
                    p = model.predict(X, verbose=0)
                # 모델별로 모든 프레임쌍의 라벨을 뽑아 빈도수 확인 -> 그 모델의 세그먼트 라벨
                frame_labels = np.argmax(p, axis=1)
                seg_label_for_this_model = Counter(frame_labels).most_common(1)[0][0]
                model_preds.append(seg_label_for_this_model)

            # 모델별 세그먼트 라벨에 대해 다시 다수결
            final_seg_label = Counter(model_preds).most_common(1)[0][0]
            segment_labels.append(final_seg_label)

        # (C) 비디오 단위 최종 라벨
        if not segment_labels:
            predictions_dict[subfolder][vname] = 'invalid'
        else:
            # 세그먼트들 다수결
            final_video_label = Counter(segment_labels).most_common(1)[0][0]
            predictions_dict[subfolder][vname] = int(final_video_label)

    return predictions_dict, frame_predictions_list

###############################################################################
# (7-1) CFU 계산 함수: frame summary 기반 CFU 산출
###############################################################################
def compute_cfu(frame_summary, weights=[0, 10, 100, 1000, 10000]):
    """
    frame_summary: DataFrame with columns ['Folder', 'Pred0', 'Pred1', 'Pred2', 'Pred3', 'Pred4', 'TotalFrames']
    weights: 각 클래스별 가중치 [0, 10, 100, 1000, 10000]
    
    각 Folder(=subfolder)마다:
    1) 가장 많은 프레임 수를 가진 클래스(max_class), 그 수(max_count)
    2) max_class와 인접한 클래스 중 프레임 수가 큰 neighbor_class, neighbor_count
    3) CFU = ((max_count * weight(max_class)) + (neighbor_count * weight(neighbor_class))) / (max_count + neighbor_count)
    
    - 반환: CFU 계산 결과를 담은 DataFrame
    """
    cfu_results = []
    for idx, row in frame_summary.iterrows():
        folder = row['Folder']
        # 각 클래스(0~4)에 대한 예측 프레임 수
        counts = [row[f"Pred{i}"] for i in range(5)]
        
        if sum(counts) == 0:
            # 프레임이 전혀 없는 경우
            cfu = None
        else:
            max_class = np.argmax(counts)
            max_count = counts[max_class]

            # 가장자리에 있는 경우 neighbor_class 선택
            if max_class == 0:
                neighbor_class = 1
            elif max_class == 4:
                neighbor_class = 3
            else:
                left = counts[max_class - 1]
                right = counts[max_class + 1]
                # 양 옆 중 더 많은 쪽
                neighbor_class = max_class - 1 if left >= right else max_class + 1
            neighbor_count = counts[neighbor_class]
            
            # CFU 계산
            numerator = (max_count * weights[max_class]) + (neighbor_count * weights[neighbor_class])
            denominator = max_count + neighbor_count
            cfu = numerator / denominator

        cfu_results.append({'Folder': folder, 'CFU': cfu})
    
    return pd.DataFrame(cfu_results)

###############################################################################
# (6) 메인 함수: 결과를 엑셀로 저장 & NTU 계산 추가
###############################################################################
def main_no_gt():
    """
    Ground Truth(실제 라벨) 없이, 안정 프레임 기반으로 프레임/세그먼트/비디오 레벨 예측을 수행하고
    그 결과(요약, 디테일, 프레임별, CFU, NTU 등)를 엑셀 파일에 저장합니다.
    """
    # (A) 인자 파싱
    args = parse_arguments()
    os.makedirs(args.excel_dir, exist_ok=True)
    excel_path = os.path.join(args.excel_dir, "results_no_gt.xlsx")

    # (1) 폴더/영상 구조 파악 및 안정프레임 세그먼트화
    structure, video_frames_dict, all_videos_list = build_structure(
        base_path=args.base_path,
        subfolders=args.subfolders,
        stability_threshold=args.stability_threshold,
        buffer_size=args.buffer_size,
        apply_equalization=args.equalize
    )
    print(f"[INFO] Found total {len(all_videos_list)} (subfolder, video) pairs.")

    # (2) 모델 로딩 (tflite 또는 h5)
    model_files = sorted([f for f in os.listdir(args.models_dir) if f.endswith('.tflite') or f.endswith('.h5')])
    if not model_files:
        raise ValueError("No model files found in models_dir.")
    models = []

    for mf in model_files:
        mp = os.path.join(args.models_dir, mf)
        if args.model_type == 'tflite' and mf.endswith('.tflite'):
            # TFLite 모델
            interpreter = tf.lite.Interpreter(model_path=mp)
            interpreter.allocate_tensors()
            models.append(interpreter)
            print(f"[INFO] Loading model: {mp}")
        elif args.model_type == 'h5' and mf.endswith('.h5'):
            # Keras(.h5) 모델
            from tensorflow.keras.models import load_model
            loaded = load_model(mp, custom_objects={'DistillationLoss': DistillationLoss})
            models.append(loaded)
            print(f"[INFO] Loading model: {mp}")

    # (3) 영상/프레임 예측 (앙상블)
    predictions_dict, frame_predictions_list = evaluate_video_level_ensemble_no_gt(
        structure, video_frames_dict, all_videos_list,
        models, img_roi=args.img_roi,
        precision=args.precision
    )

    # (4) 폴더별(=subfolder) 영상 요약 테이블
    #     각 폴더 내 비디오의 예측 라벨 분포
    summary_rows = []
    num_classes = 5
    for subf in args.subfolders:
        if subf not in predictions_dict:
            # 해당 폴더가 structure에 없거나 invalid한 경우
            folder_dict = {f"Pred{i}": 0 for i in range(num_classes)}
            folder_dict["Invalid"] = 0
            folder_dict["Folder"] = subf
            folder_dict["TotalVideos"] = 0
            summary_rows.append(folder_dict)
            continue

        vmap = predictions_dict[subf]
        pred_counts = {i: 0 for i in range(num_classes)}
        pred_inv = 0
        # 비디오 단위 예측 라벨
        for vname, label in vmap.items():
            if isinstance(label, (int, np.integer)):
                pred_counts[int(label)] += 1
            else:
                pred_inv += 1

        total_vids = sum(pred_counts.values()) + pred_inv
        folder_dict = {f"Pred{i}": pred_counts[i] for i in range(num_classes)}
        folder_dict["Invalid"] = pred_inv
        folder_dict["Folder"] = subf
        folder_dict["TotalVideos"] = total_vids
        summary_rows.append(folder_dict)

    df_summary = pd.DataFrame(summary_rows)
    df_summary = df_summary[['Folder', 'Pred0', 'Pred1', 'Pred2', 'Pred3', 'Pred4', 'Invalid', 'TotalVideos']]

    # (5) 영상 단위 상세 테이블 (Detail)
    #     (subfolder, video, 예측라벨)
    detail_rows = []
    for subf, vdict in predictions_dict.items():
        for vname, label in vdict.items():
            detail_rows.append({
                'Folder': subf,
                'Video': vname,
                'Prediction': label
            })
    df_detail = pd.DataFrame(detail_rows)

    # (6) 프레임 단위 상세 테이블 (FrameDetail)
    df_frame_detail = pd.DataFrame(frame_predictions_list)

    # (7) 폴더 단위 프레임 예측 요약 (FrameSummary)
    #     - Folder 별로 Prediction(0~4) 별 프레임 수 집계
    if not df_frame_detail.empty:
        df_frame_summary = pd.pivot_table(
            df_frame_detail, 
            index='Folder', 
            columns='Prediction', 
            aggfunc='size', 
            fill_value=0
        )
        # 컬럼명을 Pred0, Pred1,... 로 변환
        df_frame_summary = df_frame_summary.rename(
            columns={i: f"Pred{i}" for i in range(num_classes)}
        ).reset_index()
        
        # 폴더 컬럼을 맨 앞으로, 없는 컬럼은 0으로 채움
        class_cols = [f"Pred{i}" for i in range(num_classes)]
        df_frame_summary = df_frame_summary.reindex(columns=['Folder'] + class_cols, fill_value=0)
        df_frame_summary['TotalFrames'] = df_frame_summary[class_cols].sum(axis=1)
    else:
        # 프레임 정보가 없는 경우
        df_frame_summary = pd.DataFrame(columns=['Folder'] + [f"Pred{i}" for i in range(num_classes)] + ['TotalFrames'])

    # (7-2) 각 폴더별 CFU 계산 (frame summary 기반)
    df_cfu = compute_cfu(df_frame_summary)

    # (8) NTU (탁도) 계산
    #     각 영상 단위로 자기상관(autocor) -> 최대값 -> calculate_turbidity
    turbidity_results = []
    for key, frames in video_frames_dict.items():
        subfolder, vname = key
        if frames.shape[0] < 2:
            continue
        # 1-시차 자기상관
        acor_values = autocor(frames, n=1)
        max_acor = np.max(acor_values)
        # NTU = a*acor^3 + b*acor^2 + c*acor + d
        ntu = calculate_turbidity(max_acor, args.a, args.b, args.c, args.d)
        turbidity_results.append({'Subfolder': subfolder, 'Video': vname, 'NTU': f"{ntu:.3f}"})

    df_ntu = pd.DataFrame(turbidity_results)
    if not df_ntu.empty:
        avg_ntu = df_ntu['NTU'].astype(float).mean()
        df_avg = pd.DataFrame([{'Subfolder': 'Average', 'Video': '', 'NTU': f"{avg_ntu:.3f}"}])
        df_ntu = pd.concat([df_ntu, df_avg], ignore_index=True)

    # 결과 데이터프레임들 출력
    print("\n[Summary Table]")
    print(df_summary)
    print("\n[FrameSummary Table]")
    print(df_frame_summary)
    print("\n[CFU Table]")
    print(df_cfu)
    print("\n[NTU Table]")
    print(df_ntu)

    # (9) 엑셀 저장 (NTU 결과 시트 포함)
    os.makedirs(args.excel_dir, exist_ok=True)
    with pd.ExcelWriter(excel_path) as writer:
        df_summary.to_excel(writer, sheet_name='Summary', index=False)           # 폴더별 비디오 예측 요약
        df_detail.to_excel(writer, sheet_name='Detail', index=False)            # 비디오 단위 결과
        df_frame_detail.to_excel(writer, sheet_name='FrameDetail', index=False) # 프레임 단위 결과
        df_frame_summary.to_excel(writer, sheet_name='FrameSummary', index=False)
        df_cfu.to_excel(writer, sheet_name='CFU', index=False)                  # CFU 계산 결과
        df_ntu.to_excel(writer, sheet_name='NTU', index=False)                  # NTU 계산 결과

    print(f"\n[Done] Saved to: {excel_path}")
    print("Check 'Summary' for folder-level distribution (videos).")
    print("Check 'Detail' for each video-level prediction.")
    print("Check 'FrameDetail' for each frame-pair prediction.")
    print("Check 'FrameSummary' for folder-level frame distribution.")
    print("Check 'CFU' for CFU calculation results.")
    print("Check 'NTU' for NTU (turbidity) calculation results.")

###############################################################################
# 사용자 정의 손실 함수: DistillationLoss
#   - Knowledge Distillation을 위한 예시
###############################################################################
class DistillationLoss(losses.Loss):
    """
    Knowledge Distillation용 사용자 정의 손실 함수.
    - teacher_predictions: Teacher 모델의 예측(로그잇) 값
    - temperature: 소프트닝 온도
    - alpha: Distillation loss와 crossentropy 간 가중치 비율
    """
    def __init__(self, teacher_predictions=None, temperature=3, alpha=0.1, 
                 reduction=losses.Reduction.AUTO, name='distillation_loss', **kwargs):
        super(DistillationLoss, self).__init__(reduction=reduction, name=name, **kwargs)
        self.teacher_predictions = teacher_predictions
        self.temperature = temperature
        self.alpha = alpha

    def call(self, y_true, y_pred):
        # Teacher 없이 일반 학습
        if self.teacher_predictions is None:
            return losses.categorical_crossentropy(y_true, y_pred)
        else:
            batch_size = tf.shape(y_pred)[0]
            teacher_pred_batch = tf.slice(self.teacher_predictions, [0, 0], [batch_size, -1])
            # Teacher/Student 모델 예측(로그잇)에 temperature를 적용하여 soft label
            soft_teacher_pred = tf.nn.softmax(teacher_pred_batch / self.temperature)
            soft_student_pred = tf.nn.softmax(y_pred / self.temperature)

            distillation_loss = losses.KLDivergence()(soft_teacher_pred, soft_student_pred)
            hard_loss = losses.categorical_crossentropy(y_true, y_pred)
            return self.alpha * distillation_loss + (1 - self.alpha) * hard_loss


if __name__ == "__main__":
    # 엔트리 포인트
    main_no_gt()
