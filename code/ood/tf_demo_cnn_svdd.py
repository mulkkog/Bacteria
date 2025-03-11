import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import Counter
import pandas as pd
from PIL import Image
from skimage import exposure
from scipy.signal import find_peaks
from sklearn.metrics import confusion_matrix

###############################################################################
# 1) 인자 파싱
###############################################################################
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Demo to test One-Class(Deep SVDD) fold models on new data, with invalid-video info."
    )
    parser.add_argument('--base_path', type=str,
                        default='dataset/2025_bacteria/250121',
                        help='Single date folder path with subfolders=normal(0)/anomaly(1).')
    parser.add_argument('--subfolders', type=str, nargs='+', default=['0','1'],
                        help='Class subfolders, e.g. 0 1 ... for evaluation reference.')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='For confusion matrix indexing (0=normal, 1=anomaly).')
    parser.add_argument('--models_dir', type=str, default='models/stable_svdd',
                        help='Directory with fold_n_model.h5, fold_n_center.npy, fold_n_threshold.txt, etc.')
    parser.add_argument('--excel_dir', type=str, default='excel/2025_bacteria/250121_svdd',
                        help='Where to save the results Excel.')
    parser.add_argument('--img_roi', type=int, default=96,
                        help='Crop region of interest size.')
    parser.add_argument('--stability_threshold', type=int, default=350)
    parser.add_argument('--buffer_size', type=int, default=2)
    parser.add_argument('--equalize', action='store_true')
    parser.add_argument('--use_ensemble', action='store_false',
                        help='If multiple fold models exist, also evaluate an ensemble (majority vote).')
    return parser.parse_args()

###############################################################################
# 2) 이미지 로딩 (TIFF / YUV)
###############################################################################
def load_image(file_path, apply_equalization):
    if file_path.lower().endswith('.tiff'):
        with Image.open(file_path) as img:
            arr = np.array(img, dtype=np.float32) 
        # 16비트 TIFF 가능성 처리:
        if arr.max() > 1.0:  
            arr /= 65535.0 if arr.max() > 255 else 255.0
    elif file_path.lower().endswith('.yuv'):
        with open(file_path,'rb') as f:
            raw = f.read(128*128)
            arr = np.frombuffer(raw, dtype=np.uint8).reshape((128,128))/255.0
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    if apply_equalization:
        arr = exposure.equalize_hist(arr)
    return arr.astype(np.float32)

###############################################################################
# 3) 안정 프레임(stable frames) 찾기
###############################################################################
def find_stable_frames(images, stability_threshold, buffer_size):
    if images.shape[0] < 2:
        return []
    diffs = np.abs(images[:-1] - images[1:])
    summed = diffs.sum(axis=(1,2))
    peaks, _ = find_peaks(summed, height=stability_threshold)

    excluded = set()
    for peak in peaks:
        start = max(0, peak - buffer_size)
        end = min(len(summed), peak + buffer_size + 1)
        for i in range(start, end):
            excluded.add(i)
    stable = [i for i in range(len(summed)) if i not in excluded]
    return stable

###############################################################################
# 4) 폴더 구조 분석 + 영상 로딩
###############################################################################
def build_structure(base_path, subfolders, stability_threshold, buffer_size, apply_equalization):
    structure = {}
    video_frames_dict = {}
    all_videos_list = []
    
    for subfolder in subfolders:
        folder_path = os.path.join(base_path, subfolder)
        if not os.path.isdir(folder_path):
            continue

        structure[subfolder] = {}
        for vdir in sorted(os.listdir(folder_path)):
            video_path = os.path.join(folder_path, vdir)
            if not os.path.isdir(video_path):
                continue

            all_videos_list.append((subfolder, vdir))

            frame_files = sorted([
                f for f in os.listdir(video_path)
                if f.lower().endswith('.tiff') or f.lower().endswith('.yuv')
            ])
            if not frame_files:
                continue

            frames = []
            for ff in frame_files:
                fp = os.path.join(video_path, ff)
                arr = load_image(fp, apply_equalization)
                frames.append(arr)
            frames = np.array(frames, dtype=np.float32)

            stable = find_stable_frames(frames, stability_threshold, buffer_size)
            if not stable:
                # stable 없으면 세그먼트가 없다
                continue

            seg_dict = {}
            seg_id = 0
            start = stable[0]
            for i in range(1, len(stable)):
                if stable[i] != stable[i-1] + 1:
                    end = stable[i-1] + 1
                    seg_dict[f"segment_{seg_id}"] = list(range(start,end))
                    seg_id += 1
                    start = stable[i]
            end = stable[-1] + 1
            seg_dict[f"segment_{seg_id}"] = list(range(start, end))

            structure[subfolder][vdir] = seg_dict
            video_frames_dict[(subfolder, vdir)] = frames

    return structure, video_frames_dict, all_videos_list

###############################################################################
# 5) 프레임 레벨 데이터셋
###############################################################################
def build_frame_level_dataset(structure, video_frames_dict, img_roi):
    X_list = []
    Y_list = []
    for subfolder, videos in structure.items():
        true_label = int(subfolder)  # 0 or 1
        for vname, segs in videos.items():
            frames = video_frames_dict.get((subfolder, vname), None)
            if frames is None:
                continue
            for seg_id, idx_list in segs.items():
                if len(idx_list) <= 1:
                    continue
                seg_frames = frames[idx_list]
                diffs = np.abs(seg_frames[:-1] - seg_frames[1:])
                if diffs.shape[0]==0:
                    continue
                diffs = diffs[:, :img_roi, :img_roi]
                X_list.append(diffs)
                Y_list.extend([true_label]*diffs.shape[0])

    if not X_list:
        return None, None
    X_data = np.concatenate(X_list, axis=0)
    X_data = np.expand_dims(X_data, axis=-1)
    y_data = np.array(Y_list, dtype=int)
    return X_data, y_data

###############################################################################
# 6) One-Class 모델(embedding + center + threshold)로 0 or 1 예측
###############################################################################
def get_distance_prediction(model, center, threshold, X_batch):
    emb = model(X_batch, training=False).numpy()
    diff = emb - center
    dist = np.sqrt(np.sum(diff**2, axis=1))
    y_pred = (dist > threshold).astype(int)
    return y_pred

def evaluate_frame_single_oneclass(model, center, threshold, X_test, y_test):
    if X_test is None or len(X_test)==0:
        return 0.0, np.zeros((2,2), dtype=int)
    y_pred = get_distance_prediction(model, center, threshold, X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0,1])
    acc = np.mean(y_test == y_pred)
    return acc, cm

def evaluate_frame_ensemble_oneclass(model_info_list, X_test, y_test):
    if X_test is None or len(X_test)==0:
        return 0.0, np.zeros((2,2), dtype=int)

    all_preds = []
    for (m, c, thr) in model_info_list:
        preds = get_distance_prediction(m, c, thr, X_test)
        all_preds.append(preds.reshape(-1,1))
    all_preds = np.hstack(all_preds)
    final_preds = []
    for row in all_preds:
        mc = Counter(row).most_common(1)[0][0]
        final_preds.append(mc)
    final_preds = np.array(final_preds, dtype=int)

    cm = confusion_matrix(y_test, final_preds, labels=[0,1])
    acc = np.mean(y_test == final_preds)
    return acc, cm

###############################################################################
# 7) 비디오 레벨 평가 + invalid 정보
###############################################################################
def evaluate_video_level_single_oneclass(
    structure, video_frames_dict, all_videos_list,
    model, center, threshold, img_roi
):
    """
    Return: conf_mat(3x3), invalid_list
     - invalid_list: [(subfolder,video_name,reason), ...]
    """
    num_cl = 2
    invalid_idx = 2
    conf_mat = np.zeros((num_cl+1, num_cl+1), dtype=int)
    invalid_info = []

    # (A) 세그먼트별 라벨
    video_seg_labels = {}  
    for sb, video_map in structure.items():
        tlabel = int(sb)
        for vname, seg_dict in video_map.items():
            frames = video_frames_dict.get((sb, vname), None)
            if frames is None:
                # 파일 없었나
                continue

            for seg_id, idx_list in seg_dict.items():
                if len(idx_list)<=1:
                    continue
                seg_frames = frames[idx_list]
                diffs = np.abs(seg_frames[:-1] - seg_frames[1:])
                if diffs.shape[0]==0:
                    continue
                diffs = np.expand_dims(diffs[:, :img_roi, :img_roi], axis=-1)

                seg_pred = get_distance_prediction(model, center, threshold, diffs)
                seg_label = Counter(seg_pred).most_common(1)[0][0]
                video_seg_labels.setdefault((tlabel, vname), []).append(seg_label)

    # (B) 비디오 레벨 라벨 (세그먼트가 여러 개이면 majority)
    assigned_videos = set()
    for (tlabel, vname), seg_labels in video_seg_labels.items():
        final_label = Counter(seg_labels).most_common(1)[0][0]
        conf_mat[tlabel, final_label] += 1
        assigned_videos.add((tlabel, vname))

    # (C) invalid 처리 
    #     all_videos_list 중 assigned되지 않은 비디오 => invalid
    for (sb, vname) in all_videos_list:
        tlabel = int(sb)
        if (tlabel, vname) not in assigned_videos:
            conf_mat[tlabel, invalid_idx] += 1
            # reason 구체화
            #  1) 아예 structure에 없는 경우 => no stable frames
            #  2) structure엔 있는데 segment가 전부 <=1 => insufficient
            #     => 실제로는 (tlabel,vname)이 structure에 있을 수도, 없을 수도
            if sb not in structure or vname not in structure[sb]:
                # no stable segments
                reason = "No stable frames"
            else:
                # segment 존재하나 length<=1
                reason = "Insufficient segment length"
            invalid_info.append((sb, vname, reason))

    return conf_mat, invalid_info


def evaluate_video_level_ensemble_oneclass(
    structure, video_frames_dict, all_videos_list,
    model_info_list, img_roi
):
    """
    Return: conf_mat(3x3), invalid_list
    """
    num_cl = 2
    invalid_idx = 2
    conf_mat = np.zeros((num_cl+1, num_cl+1), dtype=int)
    invalid_info = []

    video_seg_labels = {}
    for sb, video_map in structure.items():
        tlabel = int(sb)
        for vname, seg_dict in video_map.items():
            frames = video_frames_dict.get((sb, vname), None)
            if frames is None:
                continue
            for seg_id, idx_list in seg_dict.items():
                if len(idx_list)<=1:
                    continue
                seg_frames = frames[idx_list]
                diffs = np.abs(seg_frames[:-1] - seg_frames[1:])
                if diffs.shape[0]==0:
                    continue
                diffs = np.expand_dims(diffs[:, :img_roi, :img_roi], axis=-1)

                # 모델별 예측 -> majority
                seg_labels_from_models = []
                for (m, c, thr) in model_info_list:
                    sp = get_distance_prediction(m, c, thr, diffs)
                    seg_labels_from_models.append(sp)
                # shape=(num_models, nFrames)
                seg_labels_from_models = np.vstack(seg_labels_from_models).T
                # 각 frame에서 majority
                frame_preds = []
                for row in seg_labels_from_models:
                    mc = Counter(row).most_common(1)[0][0]
                    frame_preds.append(mc)
                # 이제 이 segment 라벨
                seg_label = Counter(frame_preds).most_common(1)[0][0]
                video_seg_labels.setdefault((tlabel, vname), []).append(seg_label)

    assigned_videos = set()
    for (tlabel, vname), seg_labels in video_seg_labels.items():
        final_label = Counter(seg_labels).most_common(1)[0][0]
        conf_mat[tlabel, final_label] += 1
        assigned_videos.add((tlabel, vname))

    for (sb, vname) in all_videos_list:
        tlabel = int(sb)
        if (tlabel, vname) not in assigned_videos:
            conf_mat[tlabel, invalid_idx] += 1
            if sb not in structure or vname not in structure[sb]:
                reason = "No stable frames"
            else:
                reason = "Insufficient segment length"
            invalid_info.append((sb, vname, reason))

    return conf_mat, invalid_info


###############################################################################
# 8) 메인 실행
###############################################################################
def main():
    args = parse_arguments()
    os.makedirs(args.excel_dir, exist_ok=True)
    excel_path = os.path.join(args.excel_dir, "demo_oneclass_results.xlsx")

    # A. 폴더 구조 + 영상 로딩
    structure, video_frames_dict, all_videos_list = build_structure(
        base_path=args.base_path,
        subfolders=args.subfolders,
        stability_threshold=args.stability_threshold,
        buffer_size=args.buffer_size,
        apply_equalization=args.equalize
    )
    print(f"[INFO] Found {len(all_videos_list)} (subfolder,video) combos (physical videos).")

    # B. 프레임 레벨 데이터
    X_test, y_test = build_frame_level_dataset(
        structure, video_frames_dict, args.img_roi
    )

    # C. One-Class 모델(들) 로드 
    model_files = sorted([f for f in os.listdir(args.models_dir) if f.endswith('_model.h5')])
    if not model_files:
        # 폴드명 안붙은 모델만 있을 수도?
        default_h5 = [f for f in os.listdir(args.models_dir) if f.endswith('.h5')]
        model_files = sorted(default_h5)

    if not model_files:
        raise ValueError("No One-Class .h5 model found in models_dir.")

    model_info_list = []
    for mf in model_files:
        fold_name = mf.replace('_model.h5','')
        model_path = os.path.join(args.models_dir, mf)
        center_path = os.path.join(args.models_dir, fold_name + '_center.npy')
        thr_path    = os.path.join(args.models_dir, fold_name + '_threshold.txt')

        if not (os.path.exists(center_path) and os.path.exists(thr_path)):
            print(f"[WARN] Missing center or threshold for {mf}. Skipping.")
            continue

        print(f"Loading One-Class Model: {model_path}")
        model = load_model(model_path, compile=False)
        center = np.load(center_path)
        with open(thr_path, 'r') as f:
            threshold = float(f.read().strip())
        model_info_list.append((fold_name, model, center, threshold))

    if not model_info_list:
        raise ValueError("No valid fold model with center+threshold found.")

    # D. 프레임 레벨 평가
    frame_rows = []
    for (fold_name, model, center, threshold) in model_info_list:
        facc, fcm = evaluate_frame_single_oneclass(model, center, threshold, X_test, y_test)
        frame_rows.append((fold_name, facc, fcm))

    ensemble_frame_acc, ensemble_frame_cm = (None, None)
    if args.use_ensemble and len(model_info_list) > 1:
        # (model, center, threshold)만 추출
        mct_list = [(m,c,t) for (_,m,c,t) in model_info_list]
        eacc, ecm = evaluate_frame_ensemble_oneclass(mct_list, X_test, y_test)
        ensemble_frame_acc, ensemble_frame_cm = eacc, ecm

    # E. 비디오 레벨 평가
    #    invalid 정보도 함께
    video_rows = []
    invalid_dicts = []  # 여기에 (ModelName, Subfolder, VideoName, Reason)
    for (fold_name, model, center, threshold) in model_info_list:
        vcm, invalid_info = evaluate_video_level_single_oneclass(
            structure, video_frames_dict, all_videos_list,
            model, center, threshold, args.img_roi
        )
        total_v = vcm.sum()
        v_acc = np.trace(vcm)/total_v if total_v>0 else 0
        video_rows.append((fold_name, v_acc, vcm))

        for (sb,vn,reason) in invalid_info:
            invalid_dicts.append({
                'ModelName': fold_name,
                'Subfolder': sb,
                'VideoName': vn,
                'Reason': reason
            })

    ensemble_video_acc, ensemble_video_cm = (None, None)
    invalid_info_ens = []
    if args.use_ensemble and len(model_info_list) > 1:
        mct_list = [(m,c,t) for (_,m,c,t) in model_info_list]
        vcm_e, invalid_e = evaluate_video_level_ensemble_oneclass(
            structure, video_frames_dict, all_videos_list,
            mct_list, args.img_roi
        )
        total_e = vcm_e.sum()
        v_acc_e = np.trace(vcm_e)/total_e if total_e>0 else 0
        ensemble_video_acc, ensemble_video_cm = v_acc_e, vcm_e

        for (sb,vn,reason) in invalid_e:
            invalid_dicts.append({
                'ModelName': 'Ensemble',
                'Subfolder': sb,
                'VideoName': vn,
                'Reason': reason
            })

    # F. 엑셀에 결과 저장
    frame_dicts = []
    for (fold_name, facc, fcm) in frame_rows:
        frame_dicts.append({
            'ModelName': fold_name,
            'FrameAcc': facc,
            'FrameConfMat': fcm.tolist()
        })
    if ensemble_frame_acc is not None:
        frame_dicts.append({
            'ModelName': 'Ensemble',
            'FrameAcc': ensemble_frame_acc,
            'FrameConfMat': ensemble_frame_cm.tolist() if ensemble_frame_cm is not None else None
        })
    df_frame = pd.DataFrame(frame_dicts)

    video_dicts = []
    for (fold_name, vacc, vcm) in video_rows:
        video_dicts.append({
            'ModelName': fold_name,
            'VideoAcc': vacc,
            'VideoConfMat': vcm.tolist()
        })
    if ensemble_video_acc is not None:
        video_dicts.append({
            'ModelName': 'Ensemble',
            'VideoAcc': ensemble_video_acc,
            'VideoConfMat': ensemble_video_cm.tolist() if ensemble_video_cm is not None else None
        })
    df_video = pd.DataFrame(video_dicts)

    df_invalid = pd.DataFrame(invalid_dicts)

    with pd.ExcelWriter(excel_path) as writer:
        df_frame.to_excel(writer, sheet_name='FrameLevel', index=False)
        df_video.to_excel(writer, sheet_name='VideoLevel', index=False)
        df_invalid.to_excel(writer, sheet_name='InvalidVideos', index=False)

    print(f"\n[Done] One-Class testing results saved to {excel_path}")
    print(f"Frame-level sheet: 'FrameLevel'; Video-level sheet: 'VideoLevel'; Invalid sheet: 'InvalidVideos'.\n")

if __name__ == "__main__":
    main()
