import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import pandas as pd
from tensorflow.keras import backend as K
from spektral.layers import GCNConv, GlobalAvgPool
from spektral.data import Dataset, Graph
from spektral.data.loaders import BatchLoader
from collections import Counter, defaultdict
from scipy.signal import find_peaks
from PIL import Image
from skimage import exposure
import gc
import imageio  # imageio 사용

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

# ----------------------------------------------
# 1. 시드 고정
# ----------------------------------------------
def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(42)

# ----------------------------------------------
# 2. GCN 모델 정의
# ----------------------------------------------
def create_gcn_model(num_patches, num_features, num_classes):
    x_input = tf.keras.Input(shape=(num_patches, num_features), name='X_input')
    a_input = tf.keras.Input(shape=(num_patches, num_patches), sparse=True, name='A_input')

    x = GCNConv(32, activation='relu')([x_input, a_input])
    x = GCNConv(64, activation='relu')([x, a_input])
    x = GlobalAvgPool()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=[x_input, a_input], outputs=outputs)
    return model

# ----------------------------------------------
# 3. 이미지 -> 그래프 유틸
# ----------------------------------------------
def image_to_patches(image, patch_size):
    patches = []
    for i in range(0, image.shape[0], patch_size):
        for j in range(0, image.shape[1], patch_size):
            patch = image[i:i+patch_size, j:j+patch_size]
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                patches.append(patch)
    return np.array(patches)

def create_grid_adjacency(patches_per_dim_height, patches_per_dim_width):
    num_nodes = patches_per_dim_height * patches_per_dim_width
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(patches_per_dim_height):
        for j in range(patches_per_dim_width):
            node = i * patches_per_dim_width + j
            # 상하좌우 연결
            if i > 0:
                adj[node, (i-1)*patches_per_dim_width + j] = 1
            if i < patches_per_dim_height - 1:
                adj[node, (i+1)*patches_per_dim_width + j] = 1
            if j > 0:
                adj[node, i*patches_per_dim_width + (j-1)] = 1
            if j < patches_per_dim_width - 1:
                adj[node, i*patches_per_dim_width + (j+1)] = 1
    return adj

def load_tiff_image(file_path, apply_equalization):
    image = imageio.imread(file_path)
    # 16-bit / 8-bit 처리
    if image.dtype == np.uint16:
        image = image.astype(np.float32) / 65535.0
    elif image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)

    if apply_equalization:
        image = exposure.equalize_hist(image)
    return image

def load_yuv_image(file_path, apply_equalization=False):
    file_size = os.path.getsize(file_path)
    required_size = 128 * 128
    if file_size < required_size:
        raise ValueError(f"File '{file_path}' size ({file_size} bytes) < required ({required_size} bytes).")

    with open(file_path, 'rb') as f:
        raw_data = f.read(required_size)
    if len(raw_data) != required_size:
        raise ValueError(f"Read data size ({len(raw_data)}) != required ({required_size}).")

    image = np.frombuffer(raw_data, dtype=np.uint8).reshape((128, 128))
    image = image.astype(np.float32) / 255.0
    if apply_equalization:
        image = exposure.equalize_hist(image)
    return image

def get_images(file_paths, apply_equalization):
    images = []
    for fp in file_paths:
        if fp.endswith('.tiff'):
            img = load_tiff_image(fp, apply_equalization)
        else:
            img = load_yuv_image(fp, apply_equalization)
        images.append(img)
    return np.array(images)

# ----------------------------------------------
# 4. CustomDataset (Spektral)
# ----------------------------------------------
class CustomDataset(Dataset):
    def __init__(self, graphs, labels, segment_ids, **kwargs):
        """
        graphs[i] = (X, A)
        labels[i] = 정답 라벨
        segment_ids[i] = (video_folder, segment_name) 등 식별자
        """
        self.graphs = graphs
        self.labels = labels
        self.segment_ids = segment_ids
        super().__init__(**kwargs)

    def read(self):
        output = []
        for i, (X, A) in enumerate(self.graphs):
            y = self.labels[i]
            g = Graph(x=X, a=A, y=y)
            output.append(g)
        return output

# ----------------------------------------------
# 5. 안정 세그먼트 찾는 함수 (원본 동일)
# ----------------------------------------------
def find_all_stable_segments(summed_diffs, peaks, buffer_size):
    total_frames = len(summed_diffs)
    excluded_indices = set()
    for peak in peaks:
        for i in range(max(0, peak - buffer_size), min(total_frames, peak + buffer_size + 1)):
            excluded_indices.add(i)
    valid_frames = [i for i in range(total_frames) if i not in excluded_indices]
    return valid_frames

def load_yuv_images(folder_path, file_names, apply_equalization=False):
    imgs = []
    for fn in file_names:
        fp = os.path.join(folder_path, fn)
        imgs.append(load_yuv_image(fp, apply_equalization))
    return np.array(imgs)

def load_tiff_images(folder_path, file_names, apply_equalization=False):
    imgs = []
    for fn in file_names:
        fp = os.path.join(folder_path, fn)
        imgs.append(load_tiff_image(fp, apply_equalization))
    return np.array(imgs)

def process_video_files(folder_path, file_names, stability_threshold, buffer_size, apply_equalization=False):
    if not file_names:
        return []
    if file_names[0].endswith('.yuv'):
        images = load_yuv_images(folder_path, file_names, apply_equalization)
    elif file_names[0].endswith('.tiff'):
        images = load_tiff_images(folder_path, file_names, apply_equalization)
    else:
        raise ValueError("Unsupported file format")
    image_diffs = np.abs(images[:-1] - images[1:])
    summed_diffs = image_diffs.sum(axis=(1,2))
    peaks, _ = find_peaks(summed_diffs, height=stability_threshold)
    stable_segments = find_all_stable_segments(summed_diffs, peaks, buffer_size)
    return stable_segments

# ----------------------------------------------
# 6. get_temporary_structure_with_stable_segments
#    → 1번 코드처럼 "no stable segments" & "insufficient frames" 처리 추가
# ----------------------------------------------
def get_temporary_structure_with_stable_segments(base_path, date_folders, subfolders,
                                                stability_threshold, buffer_size,
                                                apply_equalization=False):
    structure = {}
    no_stable_segments_videos = []
    insufficient_frame_videos = []  # ★ 추가

    for date_folder in date_folders:
        date_path = os.path.join(base_path, date_folder)
        if not os.path.exists(date_path):
            continue
        structure[date_folder] = {}
        for subfolder in subfolders:
            subfolder_path = os.path.join(date_path, subfolder)
            if not os.path.exists(subfolder_path):
                continue
            structure[date_folder][subfolder] = {}
            for video_name_folder in os.listdir(subfolder_path):
                video_folder_path = os.path.join(subfolder_path, video_name_folder)
                if not os.path.isdir(video_folder_path):
                    continue

                video_files = sorted([f for f in os.listdir(video_folder_path)
                                      if f.endswith('.yuv') or f.endswith('.tiff')])
                if not video_files:
                    continue

                stable_segments = process_video_files(
                    video_folder_path, video_files,
                    stability_threshold, buffer_size,
                    apply_equalization
                )
                if stable_segments:
                    # 안정 세그먼트 존재
                    structure[date_folder][subfolder][video_name_folder] = {}
                    start = stable_segments[0]
                    segment_id = 0
                    for i in range(1, len(stable_segments)):
                        if stable_segments[i] != stable_segments[i - 1] + 1:
                            end = stable_segments[i - 1] + 1
                            seg_files = [os.path.join(video_folder_path, video_files[j])
                                         for j in range(start, end)]
                            structure[date_folder][subfolder][video_name_folder][f"segment_{segment_id}"] = seg_files
                            segment_id += 1
                            start = stable_segments[i]
                    # 마지막 세그먼트
                    end = stable_segments[-1] + 1
                    seg_files = [os.path.join(video_folder_path, video_files[j])
                                 for j in range(start, end)]
                    structure[date_folder][subfolder][video_name_folder][f"segment_{segment_id}"] = seg_files

                    # 1번 코드처럼, "모든 세그먼트가 insufficient(파일이 2장 이하)" 인지 체크
                    segments_dict = structure[date_folder][subfolder][video_name_folder]
                    insufficient_count = 0
                    for seg_k, seg_files_list in segments_dict.items():
                        if seg_files_list is None or len(seg_files_list) <= 1:
                            insufficient_count += 1
                    if insufficient_count == len(segments_dict):
                        # 세그먼트들은 있는데 전부 insufficient
                        insufficient_frame_videos.append(
                            (date_folder, subfolder, video_name_folder, "Insufficient frames")
                        )
                else:
                    # 안정 세그먼트가 전혀 없음
                    no_stable_segments_videos.append(
                        (date_folder, subfolder, video_name_folder, "No stable segments")
                    )

    # 3가지 결과를 모두 반환
    return structure, no_stable_segments_videos, insufficient_frame_videos

# ----------------------------------------------
# 7. 이미지 → 그래프 (프레임) 구성
#    len(file_paths) <= 1이면 skip (insufficient)
# ----------------------------------------------
def process_images_to_graph(structure, num_classes, apply_equalization, patch_size=32):
    graphs = []
    labels = []
    segment_ids = []

    for date_folder, subfolders in structure.items():
        for subfolder, video_folders in subfolders.items():
            for video_folder, segments in video_folders.items():
                for seg_name, file_paths in segments.items():
                    if len(file_paths) <= 1:
                        # 1번 코드와 똑같이, insufficient은 건너뜀
                        continue

                    images = get_images(file_paths, apply_equalization)
                    for img in images:
                        patches = image_to_patches(img, patch_size)
                        num_patches = patches.shape[0]
                        X = patches.reshape(num_patches, -1)

                        # 가로, 세로 패치 개수
                        patches_per_dim = img.shape[0] // patch_size
                        A = create_grid_adjacency(patches_per_dim, patches_per_dim)

                        graphs.append((X, A))
                        labels.append(int(subfolder))
                        segment_ids.append((video_folder, seg_name))

    return graphs, np.array(labels), segment_ids

# ----------------------------------------------
# 8. 프레임 레벨 평가
# ----------------------------------------------
def evaluate_frame_level(model, loader, dataset, num_classes):
    y_pred_all = []
    y_true_all = []
    seg_ids_all = list(dataset.segment_ids)

    steps = loader.steps_per_epoch
    for _ in tqdm(range(steps), desc="Evaluating frames", unit="batch"):
        try:
            ((graphs_x, graphs_a), labels) = next(loader.load())
        except StopIteration:
            break

        preds = model.predict([graphs_x, graphs_a], verbose=0)
        pred_classes = np.argmax(preds, axis=1)

        y_pred_all.extend(pred_classes)
        y_true_all.extend(labels)

    y_pred_all = np.array(y_pred_all)
    y_true_all = np.array(y_true_all)

    conf_matrix = confusion_matrix(y_true_all, y_pred_all, labels=range(num_classes))
    accuracy = np.mean(y_true_all == y_pred_all)

    return accuracy, conf_matrix, y_pred_all, y_true_all, seg_ids_all

# ----------------------------------------------
# 9. 세그먼트, 비디오 레벨 집계
# ----------------------------------------------
def find_mode(lst):
    return Counter(lst).most_common(1)[0][0]

def evaluate_segment_and_video_level(y_pred_frames, y_true_frames, seg_ids, num_classes):
    seg_pred_map = defaultdict(list)
    seg_true_map = defaultdict(list)

    for pred_cls, true_cls, sid in zip(y_pred_frames, y_true_frames, seg_ids):
        seg_pred_map[sid].append(pred_cls)
        seg_true_map[sid].append(true_cls)

    # 세그먼트 레벨
    segment_preds = []
    segment_trues = []
    segment_video_folder_map = {}

    for sid in seg_pred_map.keys():
        pred_label = find_mode(seg_pred_map[sid])
        true_label = find_mode(seg_true_map[sid])  # 보통 전부 같은 subfolder일 것
        segment_preds.append(pred_label)
        segment_trues.append(true_label)
        video_folder, _ = sid
        segment_video_folder_map[sid] = video_folder

    segment_preds = np.array(segment_preds)
    segment_trues = np.array(segment_trues)

    seg_conf_matrix = confusion_matrix(segment_trues, segment_preds, labels=range(num_classes))
    seg_accuracy = np.mean(segment_trues == segment_preds)

    # 비디오 레벨
    video_pred_map = defaultdict(list)
    video_true_map = defaultdict(list)
    for sid, spred, strue in zip(seg_pred_map.keys(), segment_preds, segment_trues):
        vfolder = segment_video_folder_map[sid]
        video_pred_map[vfolder].append(spred)
        video_true_map[vfolder].append(strue)

    video_preds = []
    video_trues = []
    for vfolder in video_pred_map.keys():
        vpred = find_mode(video_pred_map[vfolder])
        vtrue = find_mode(video_true_map[vfolder])
        video_preds.append(vpred)
        video_trues.append(vtrue)

    video_preds = np.array(video_preds)
    video_trues = np.array(video_trues)

    video_conf_matrix = confusion_matrix(video_trues, video_preds, labels=range(num_classes))
    video_accuracy = np.mean(video_preds == video_trues)

    return seg_accuracy, seg_conf_matrix, video_accuracy, video_conf_matrix

# ----------------------------------------------
# 10. F1, Cutoff
# ----------------------------------------------
def calculate_f1_scores(conf_matrix):
    f1_scores = []
    num_classes = conf_matrix.shape[0]
    for i in range(num_classes):
        tp = conf_matrix[i, i]
        fp = conf_matrix[:, i].sum() - tp
        fn = conf_matrix[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision+recall) > 0 else 0
        f1_scores.append(f1)
    return f1_scores

def calculate_cutoff_accuracy(conf_matrix):
    # 예: [0,1] vs [2,3,4] 같은 식으로 왼 위 2행2열 + 오른 아래 3행3열을 구분하여 정확도 계산
    tp_top_left = np.sum(conf_matrix[0:2, 0:2])
    tp_bottom_right = np.sum(conf_matrix[2:, 2:])
    total_true_positives = tp_top_left + tp_bottom_right
    total_samples = np.sum(conf_matrix)
    acc = total_true_positives / total_samples if total_samples else 0
    return acc

# ----------------------------------------------
# 11. main
# ----------------------------------------------
def main(args):
    os.makedirs(args.excel_dir, exist_ok=True)
    excel_path = os.path.join(args.excel_dir, "test_results.xlsx")
    df_dummy = pd.DataFrame()
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_dummy.to_excel(writer)

    date_folders = sorted([d for d in os.listdir(args.base_path)
                           if os.path.isdir(os.path.join(args.base_path, d))])
    if not date_folders:
        raise ValueError("No date folders found.")

    results = []

    # ★ 추가: 전체(모든 fold) confusion matrix 누적용
    frame_cm_total = np.zeros((args.num_classes, args.num_classes), dtype=np.int32)
    seg_cm_total = np.zeros((args.num_classes, args.num_classes), dtype=np.int32)
    vid_cm_total = np.zeros((args.num_classes, args.num_classes), dtype=np.int32)
    # (cutoff은 최종적으로 동작하는지 여부가 subfolder=5개인지에 달려있으므로,
    #  여기서는 CM만 누적하고 마지막에 필요하다면 전체 CM으로 계산 가능.)

    for fold_index, test_date in enumerate(date_folders):
        print(f"\n--- Fold {fold_index+1}: Test on {test_date} ---")
        train_folders = [d for d in date_folders if d != test_date]
        val_folders = [test_date]

        # 테스트용 구조 구성
        val_structure, no_stable_segs, insufficient_frames = get_temporary_structure_with_stable_segments(
            args.base_path, val_folders, args.subfolders,
            args.stability_threshold, args.buffer_size,
            apply_equalization=args.equalize
        )
        print("No stable segments videos:", no_stable_segs)
        print("Insufficient frame videos:", insufficient_frames)

        # 그래프/라벨/세그먼트ID 구성
        X_test, y_test, seg_ids_test = process_images_to_graph(
            val_structure, args.num_classes, args.equalize, args.patch_size
        )
        if len(X_test) == 0:
            print("No test data. Skipping.")
            continue

        # 모델 로드
        model_path = os.path.join(args.models_dir, f"Fold_{fold_index+1}_model.h5")
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            continue

        model = load_model(
            model_path,
            custom_objects={'GCNConv': GCNConv, 'GlobalAvgPool': GlobalAvgPool}
        )

        # Spektral Dataset & Loader
        test_dataset = CustomDataset(X_test, y_test, seg_ids_test)
        test_loader = BatchLoader(test_dataset, batch_size=64, shuffle=False)

        # (A) 프레임 레벨 평가
        frame_acc, frame_cm, y_pred_frames, y_true_frames, seg_ids_frames = evaluate_frame_level(
            model, test_loader, test_dataset, args.num_classes
        )

        # (B) 세그먼트 & 비디오 레벨
        seg_acc, seg_cm, vid_acc, vid_cm = evaluate_segment_and_video_level(
            y_pred_frames, y_true_frames, seg_ids_frames, args.num_classes
        )

        # (C) F1 계산
        frame_f1 = calculate_f1_scores(frame_cm)
        seg_f1 = calculate_f1_scores(seg_cm)
        vid_f1 = calculate_f1_scores(vid_cm)

        # (D) Cutoff (subfolder==5개일 때만)
        if len(args.subfolders) == 5:
            cutoff_acc = calculate_cutoff_accuracy(vid_cm)
        else:
            cutoff_acc = None

        print("Frame Confusion Matrix:\n", frame_cm)
        print(f"Frame Accuracy: {frame_acc*100:.2f}%")
        print("Segment Confusion Matrix:\n", seg_cm)
        print(f"Segment Accuracy: {seg_acc*100:.2f}%")
        print("Video Confusion Matrix:\n", vid_cm)
        print(f"Video Accuracy: {vid_acc*100:.2f}%")
        if cutoff_acc is not None:
            print(f"Cut-off 10^2 CFU/ml: {cutoff_acc*100:.2f}%")

        # ★ 추가: fold별 CM을 누적
        frame_cm_total += frame_cm
        seg_cm_total += seg_cm
        vid_cm_total += vid_cm

        # 결과 저장
        fold_result = {
            'Fold': fold_index+1,
            'Test Date': test_date,
            'Frame CM': frame_cm.tolist(),
            'Frame Acc': frame_acc,
            'Frame F1': frame_f1,
            'Segment CM': seg_cm.tolist(),
            'Segment Acc': seg_acc,
            'Segment F1': seg_f1,
            'Video CM': vid_cm.tolist(),
            'Video Acc': vid_acc,
            'Video F1': vid_f1,
            'Cut-off 10^2 CFU/ml': cutoff_acc if cutoff_acc is not None else "N/A",
            'No stable segments': no_stable_segs,
            'Insufficient frames': insufficient_frames
        }
        results.append(fold_result)

        K.clear_session()
        del model
        gc.collect()

    # ★ 추가: 모든 fold 끝난 후, 누적 confusion matrix로 전체 성능 계산
    # -----------------------------------------------------------
    print("\n=== Overall Results (All Folds Combined) ===")

    # 1) Frame-level
    overall_frame_acc = np.trace(frame_cm_total) / np.sum(frame_cm_total) if np.sum(frame_cm_total) != 0 else 0
    overall_frame_f1 = calculate_f1_scores(frame_cm_total)
    print("Overall Frame Confusion Matrix:\n", frame_cm_total)
    print(f"Overall Frame Accuracy: {overall_frame_acc * 100:.2f}%")
    print(f"Overall Frame F1: {overall_frame_f1}")

    # 2) Segment-level
    overall_seg_acc = np.trace(seg_cm_total) / np.sum(seg_cm_total) if np.sum(seg_cm_total) != 0 else 0
    overall_seg_f1 = calculate_f1_scores(seg_cm_total)
    print("Overall Segment Confusion Matrix:\n", seg_cm_total)
    print(f"Overall Segment Accuracy: {overall_seg_acc * 100:.2f}%")
    print(f"Overall Segment F1: {overall_seg_f1}")

    # 3) Video-level
    overall_vid_acc = np.trace(vid_cm_total) / np.sum(vid_cm_total) if np.sum(vid_cm_total) != 0 else 0
    overall_vid_f1 = calculate_f1_scores(vid_cm_total)
    print("Overall Video Confusion Matrix:\n", vid_cm_total)
    print(f"Overall Video Accuracy: {overall_vid_acc * 100:.2f}%")
    print(f"Overall Video F1: {overall_vid_f1}")
    # (Cutoff도 원한다면, 여기서 한번에 계산할 수도 있음.)

    # Excel 저장
    df = pd.DataFrame(results)
    df.to_excel(excel_path, index=False)
    print(f"Done. Results saved to: {excel_path}")

# ----------------------------------------------
# 12. 인자 파싱
# ----------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="Script for GCN evaluation (frame -> segment -> video).")
    parser.add_argument('--base_path', type=str,
                        default='/home/jijang/ssd_data/projects/Bacteria/dataset/stable')
    parser.add_argument('--models_dir', type=str,
                        default='/home/jijang/ssd_data/projects/Bacteria/models/stable_gnn')
    parser.add_argument('--excel_dir', type=str,
                        default='/home/jijang/ssd_data/projects/Bacteria/excel/stable_gnn')
    parser.add_argument('--subfolders', type=str, nargs='+',
                        default=['0','1'])
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--stability_threshold', type=int, default=350)
    parser.add_argument('--buffer_size', type=int, default=2)
    parser.add_argument('--equalize', action='store_true')
    return parser.parse_args()

# ----------------------------------------------
# 13. 실행
# ----------------------------------------------
if __name__ == "__main__":
    args = parse_arguments()
    main(args)
