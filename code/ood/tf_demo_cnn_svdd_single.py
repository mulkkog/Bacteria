import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import Counter
import pandas as pd
import imageio
from skimage import exposure
from scipy.signal import find_peaks

###############################################################################
# 1) 인자 파싱
###############################################################################
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Test script for One-Class(Deep SVDD) with multiple folds (ensemble)."
    )
    parser.add_argument('--base_path', type=str,
                        default='dataset/2025_bacteria/250121/0121-3',
                        help='Root path containing subfolders (0,P1,P2,P3, etc.).')
    parser.add_argument('--subfolders', type=str, nargs='+',
                        default=['0','P1','P2','P3'],
                        help='Which subfolders to process.')
    parser.add_argument('--fold_models_dir', type=str,
                        default='models/stable_svdd',
                        help='Directory containing Fold_{i}_model.h5, Fold_{i}_center.npy, Fold_{i}_threshold.txt.')
    parser.add_argument('--img_roi', type=int, default=96,
                        help='ROI crop size (e.g., 128 -> 128x128).')
    parser.add_argument('--stability_threshold', type=int, default=350,
                        help='Threshold for stable frame detection.')
    parser.add_argument('--buffer_size', type=int, default=2,
                        help='Buffer around peaks for excluding stable frames.')
    parser.add_argument('--equalize', action='store_true',
                        help='Apply histogram equalization if set.')
    parser.add_argument('--excel_dir', type=str,
                        default='excel/2025_bacteria/cnn_ood/250121/0121-3',
                        help='Directory to save the Excel result.')
    return parser.parse_args()

###############################################################################
# 2) 이미지 로딩 (TIFF / YUV)
###############################################################################
def load_tiff_image(file_path, apply_equalization=False):
    img = imageio.imread(file_path)
    if img.dtype == np.uint16:
        arr = img.astype(np.float32)/65535.0
    else:
        arr = img.astype(np.float32)/255.0
    if apply_equalization:
        arr = exposure.equalize_hist(arr)
    return arr

def load_yuv_image(file_path, apply_equalization=False):
    required_size = 128*128
    with open(file_path, 'rb') as f:
        raw = f.read(required_size)
    if len(raw) != required_size:
        raise ValueError(f"File size mismatch: {file_path}")
    arr = np.frombuffer(raw, dtype=np.uint8).reshape((128,128)).astype(np.float32)/255.0
    if apply_equalization:
        arr = exposure.equalize_hist(arr)
    return arr

def load_image(file_path, apply_equalization=False):
    ext = file_path.lower().split('.')[-1]
    if ext == 'tiff':
        return load_tiff_image(file_path, apply_equalization)
    elif ext == 'yuv':
        return load_yuv_image(file_path, apply_equalization)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

###############################################################################
# 3) 안정 프레임(stable frames) 찾기
###############################################################################
def find_stable_frames(images, stability_threshold, buffer_size):
    """
    images: shape=(F,H,W)
    returns stable_indices: list of frame indices
    """
    if len(images) < 2:
        return []
    diffs = np.abs(images[:-1] - images[1:])
    summed = diffs.sum(axis=(1,2))  # shape=(F-1,)
    peaks, _ = find_peaks(summed, height=stability_threshold)

    excluded = set()
    for p in peaks:
        start = max(0, p - buffer_size)
        end = min(len(summed), p + buffer_size + 1)
        for i in range(start, end):
            excluded.add(i)

    stable_indices = [i for i in range(len(summed)) if i not in excluded]
    return stable_indices

###############################################################################
# 4) 폴더 구조 분석 + 세그먼트 구성
###############################################################################
def build_structure(base_path, subfolders, stability_threshold, buffer_size, apply_equalization):
    """
    Returns:
      structure[subfolder][video_name] = { 'segment_0': [frame_file_paths...], ... }
      all_videos_list = [(subfolder, video_name)]
    """
    structure = {}
    all_videos_list = []

    for sf in subfolders:
        sf_path = os.path.join(base_path, sf)
        if not os.path.isdir(sf_path):
            print(f"[WARN] {sf_path} is not a directory. Skipping.")
            continue

        structure[sf] = {}
        vdirs = os.listdir(sf_path)
        for vd in vdirs:
            vpath = os.path.join(sf_path, vd)
            if not os.path.isdir(vpath):
                continue

            all_videos_list.append((sf, vd))

            # 프레임 파일
            frame_files = sorted([
                f for f in os.listdir(vpath)
                if f.lower().endswith('.tiff') or f.lower().endswith('.yuv')
            ])
            if not frame_files:
                # 세그먼트 없음
                structure[sf][vd] = {}
                continue

            frames = [load_image(os.path.join(vpath, ff), apply_equalization)
                      for ff in frame_files]
            frames = np.array(frames, dtype=np.float32)

            stable_idx = find_stable_frames(frames, stability_threshold, buffer_size)
            if not stable_idx:
                structure[sf][vd] = {}
                continue

            seg_dict = {}
            seg_id = 0
            start = stable_idx[0]
            for i in range(1, len(stable_idx)):
                # 연속단절 시점
                if stable_idx[i] != stable_idx[i-1] + 1:
                    end = stable_idx[i-1] + 1
                    seg_dict[f"segment_{seg_id}"] = [
                        os.path.join(vpath, frame_files[j]) for j in range(start, end)
                    ]
                    seg_id += 1
                    start = stable_idx[i]
            # 마지막 segment
            end = stable_idx[-1] + 1
            seg_dict[f"segment_{seg_id}"] = [
                os.path.join(vpath, frame_files[j]) for j in range(start, end)
            ]

            structure[sf][vd] = seg_dict

    return structure, all_videos_list

###############################################################################
# 5) Fold 모델들 로드
###############################################################################
def load_fold_models(fold_models_dir):
    """
    fold_models_dir 내에
       Fold_1_model.h5
       Fold_1_center.npy
       Fold_1_threshold.txt
       Fold_2_model.h5
       ...
    가 있다고 가정.
    => fold_list = [
         { 'model': m1, 'center': c1, 'threshold': t1 },
         { 'model': m2, 'center': c2, 'threshold': t2 },
         ...
       ]
    """
    fold_dicts = []
    filelist = os.listdir(fold_models_dir)
    # Fold_{i}_model.h5
    # Fold_{i}_center.npy
    # Fold_{i}_threshold.txt
    # i.e. fold index
    # 방법1) "Fold_(\d+)_model.h5" 정규표현식으로 찾아서 i를 파악
    # 여기서는 간단히 "Fold_X_" prefix로 parsing
    import re
    model_regex = re.compile(r"Fold_(\d+)_model\.h5$")
    # fold별 파일 찾기
    fold_indices = {}
    for fname in filelist:
        m = model_regex.match(fname)
        if m:
            fi = int(m.group(1))
            fold_indices[fi] = True

    if not fold_indices:
        raise ValueError(f"No 'Fold_x_model.h5' found in {fold_models_dir}")

    folds_sorted = sorted(list(fold_indices.keys()))
    for fi in folds_sorted:
        model_path = os.path.join(fold_models_dir, f"Fold_{fi}_model.h5")
        center_path = os.path.join(fold_models_dir, f"Fold_{fi}_center.npy")
        thresh_path = os.path.join(fold_models_dir, f"Fold_{fi}_threshold.txt")

        if not (os.path.isfile(model_path) and
                os.path.isfile(center_path) and
                os.path.isfile(thresh_path)):
            print(f"[WARN] Some files missing for Fold {fi}. Skipping.")
            continue

        print(f"[INFO] Loading Fold {fi} => {model_path}, {center_path}, {thresh_path}")
        m = load_model(model_path)
        c = np.load(center_path).astype(np.float32)
        with open(thresh_path, 'r') as f:
            t_str = f.read().strip()
        t_val = float(t_str)

        fold_dicts.append({
            'fold_index': fi,
            'model': m,
            'center': c,
            'threshold': t_val
        })

    if not fold_dicts:
        raise ValueError("No complete fold found.")
    return fold_dicts

###############################################################################
# 6) One-Class 예측(프레임 diffs) + Fold 앙상블
###############################################################################
def get_embeddings(model, X, batch_size=64):
    embs_list = []
    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size]
        emb = model(batch, training=False)
        embs_list.append(emb.numpy())
    if embs_list:
        return np.concatenate(embs_list, axis=0)
    else:
        return np.array([])

def compute_distance(embeddings, center):
    diff = embeddings - center
    dist = np.sqrt(np.sum(diff**2, axis=1))
    return dist

def predict_segment_fold_ensemble(
    frames, fold_dicts, img_roi
):
    """
    frames: shape=(F,H,W)
    fold_dicts: list of { 'model', 'center', 'threshold' }
    return: 최종 segment 라벨(0=정상,1=이상),
            단 fold가 여러 개이므로 majority vote
            => 1) 각 fold에 대해 프레임별 라벨 -> majority => fold-level seg 라벨
            => 2) fold-level 결과 majority => 최종 seg 라벨
    """
    # 연속 프레임 차
    diffs = np.abs(frames[:-1] - frames[1:])
    if len(diffs) == 0:
        return None  # 유효X

    # ROI crop
    diffs = diffs[:, :img_roi, :img_roi]
    diffs = np.expand_dims(diffs, axis=-1)  # (N, roi, roi, 1)

    fold_seg_labels = []
    for fold_item in fold_dicts:
        model = fold_item['model']
        center = fold_item['center']
        threshold = fold_item['threshold']

        # 프레임 임베딩
        embs = get_embeddings(model, diffs)
        if len(embs) == 0:
            continue
        dist = compute_distance(embs, center)
        frame_preds = (dist > threshold).astype(int)  # 0 or 1

        # fold 내 세그먼트 라벨 = 프레임 예측 majority
        fold_label = Counter(frame_preds).most_common(1)[0][0]
        fold_seg_labels.append(fold_label)

    if not fold_seg_labels:
        return None

    # 여러 fold 결과 majority
    final_seg_label = Counter(fold_seg_labels).most_common(1)[0][0]
    return final_seg_label

def predict_video_fold_ensemble(structure, sf, vname, fold_dicts, img_roi):
    """
    세그먼트가 없으면 invalid
    세그먼트별로 fold 앙상블 => seg_label
    비디오 내 seg_label majority => final_label
    """
    if sf not in structure or vname not in structure[sf]:
        return 'invalid'
    seg_dict = structure[sf][vname]
    if not seg_dict:
        return 'invalid'

    seg_labels = []
    for seg_id, file_paths in seg_dict.items():
        if len(file_paths) < 2:
            continue
        # frames 로딩
        frames = []
        for fp in file_paths:
            ext = fp.lower().split('.')[-1]
            if ext == 'tiff':
                frm = load_tiff_image(fp, apply_equalization=False)
            elif ext == 'yuv':
                frm = load_yuv_image(fp, apply_equalization=False)
            else:
                continue
            frames.append(frm)
        frames = np.array(frames, dtype=np.float32)
        if len(frames) < 2:
            continue

        seg_label = predict_segment_fold_ensemble(frames, fold_dicts, img_roi)
        if seg_label is not None:
            seg_labels.append(seg_label)

    if not seg_labels:
        return 'invalid'
    final_video_label = Counter(seg_labels).most_common(1)[0][0]  # 0 or 1
    return final_video_label

###############################################################################
# 7) 메인 실행부: 폴더/비디오 순회 → Fold 앙상블 → 엑셀 저장
###############################################################################
def main_test_one_class_ensemble():
    args = parse_arguments()
    os.makedirs(args.excel_dir, exist_ok=True)
    excel_path = os.path.join(args.excel_dir, "test_one_class_fold_ensemble.xlsx")

    # (A) 폴더 구조 + 세그먼트
    structure, all_videos_list = build_structure(
        base_path=args.base_path,
        subfolders=args.subfolders,
        stability_threshold=args.stability_threshold,
        buffer_size=args.buffer_size,
        apply_equalization=args.equalize
    )
    print(f"[INFO] Found total {len(all_videos_list)} (subfolder, video) pairs.")

    # (B) 여러 Fold 모델 로드
    fold_dicts = load_fold_models(args.fold_models_dir)
    print(f"[INFO] Total {len(fold_dicts)} folds loaded.")

    # (C) 예측
    # predictions_dict[subfolder][video_name] = 0/1/'invalid'
    predictions_dict = {}
    for (sf, vname) in all_videos_list:
        if sf not in predictions_dict:
            predictions_dict[sf] = {}
        label = predict_video_fold_ensemble(
            structure, sf, vname,
            fold_dicts, img_roi=args.img_roi
        )
        predictions_dict[sf][vname] = label

    # (D) 폴더별 집계
    summary_rows = []
    for sf in args.subfolders:
        if sf not in predictions_dict:
            summary_rows.append({
                'Folder': sf,
                'Normal(0)': 0,
                'Anomaly(1)': 0,
                'Invalid': 0,
                'TotalVideos': 0
            })
            continue

        vmap = predictions_dict[sf]
        count_0 = 0
        count_1 = 0
        count_inv = 0
        for _, lb in vmap.items():
            if lb == 0:
                count_0 += 1
            elif lb == 1:
                count_1 += 1
            else:
                count_inv += 1
        total_v = count_0 + count_1 + count_inv
        summary_rows.append({
            'Folder': sf,
            'Normal(0)': count_0,
            'Anomaly(1)': count_1,
            'Invalid': count_inv,
            'TotalVideos': total_v
        })

    df_summary = pd.DataFrame(summary_rows)

    # 세부 정보
    detail_rows = []
    for sf, vdict in predictions_dict.items():
        for vname, lb in vdict.items():
            detail_rows.append({
                'Folder': sf,
                'Video': vname,
                'Prediction': lb
            })
    df_detail = pd.DataFrame(detail_rows)

    # (E) 엑셀로 저장
    with pd.ExcelWriter(excel_path) as writer:
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
        df_detail.to_excel(writer, sheet_name='Detail', index=False)

    print(f"[DONE] Saved to {excel_path}")
    print("[INFO] 'Summary' sheet => folder-level distribution. 'Detail' => each video label.")

###############################################################################
# 실행
###############################################################################
if __name__ == "__main__":
    main_test_one_class_ensemble()
