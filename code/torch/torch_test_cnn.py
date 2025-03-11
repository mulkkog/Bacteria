import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from scipy.signal import find_peaks
from skimage import exposure
import imageio
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ConvSmallNet as the student model
class ConvSmallNet(nn.Module):
    def __init__(self, input_channels, num_classes): 
        super(ConvSmallNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=8, kernel_size=3, stride=3, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=2, stride=2, padding=0)
        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(16, 32)  # Adjust the input size accordingly
        self.fc2 = nn.Linear(32, num_classes)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ConvNet(nn.Module):
    def __init__(self, input_channels, num_classes): 
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))
        
        # The fully connected layers will use the output size of the final pooled layer
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        
        # Calculate the flatten size dynamically
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def create_model(num_classes, model_type):
    if model_type == "small":
        return ConvSmallNet(1, num_classes)
    elif model_type == "large":
        return ConvNet(1, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def process_images(structure, img_roi, num_classes):
    X_data = []
    y_data = []

    for date_folder, subfolders in structure.items():
        for subfolder, video_folders in subfolders.items():
            for video_folder, segments in video_folders.items():
                for segment_id, file_paths in segments.items():
                    if len(file_paths) <= 1:
                        continue

                    X = get_images(file_paths)  # 이제 X는 torch.Tensor 형식입니다.

                    for _ in range(len(file_paths) - 1):
                        y_data.append(int(subfolder))  # Assuming subfolder name is the class label
                    
                    X_processed = torch.abs(X - torch.roll(X, shifts=-1, dims=0))
                    X_processed = X_processed[:-1, :, :]  # 마지막 프레임 제거
                    X_data.append(X_processed)
                    
    X_data = torch.cat(X_data, dim=0).float()
    X_data = X_data.unsqueeze(1)  # 채널 추가
    y_data = torch.tensor(y_data)
    y_data = F.one_hot(y_data, num_classes=num_classes).float()
    X_data = X_data[:, :, :img_roi, :img_roi]  # img_roi 크기로 자르기

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
    return structure
 
def get_images(file_paths):
    images = []
    for file_path in file_paths:
        with open(file_path, 'rb') as image_file:
            image = np.frombuffer(image_file.read(128 * 128), dtype=np.uint8).reshape((128, 128))
            image = np.array(image) / 255.0
            images.append(image)
    images = np.array(images)  # 리스트를 numpy ndarray로 변환
    return torch.tensor(images)  # numpy ndarray를 torch tensor로 변환

def process_video_files(folder_path, file_names, stability_threshold, buffer_size):
    if not file_names:
        print("The folder is empty. No files to process.")
        return []
    if file_names[0].endswith('.yuv'):
        images = load_yuv_images(folder_path, file_names)
    elif file_names[0].endsWith('.tiff'):
        images = load_tiff_images(folder_path, file_names)
    else:
        raise ValueError("Unsupported file format")
    image_diffs = np.abs(images[:-1] - images[1:])
    summed_diffs = image_diffs.sum(axis=(1, 2))
    peaks, _ = find_peaks(summed_diffs, height=stability_threshold)
    stable_segments = find_all_stable_segments(summed_diffs, peaks, buffer_size)
    return stable_segments

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


class torchLoader(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "input": self.X[idx],
            "target": self.y[idx]
        }

def make_data_loader(X, y, batch_size, tag=''):
    dataset = torchLoader(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True if tag == 'train' else False, num_workers=4)
    return loader

def evaluate_model(model, val_loader, device):
    model.eval()
    val_loss = 0
    correct_val = 0
    total_val = 0
    criterion = nn.CrossEntropyLoss()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            input_val = batch["input"].to(device)
            target_val = batch["target"].to(device)
            output_val = model(input_val)

            loss = criterion(output_val, target_val)
            val_loss += loss.item()

            _, predicted_val = torch.max(output_val.data, 1)
            _, target_val_class = torch.max(target_val.data, 1)

            all_preds.extend(predicted_val.cpu().numpy())
            all_targets.extend(target_val_class.cpu().numpy())

            total_val += target_val.size(0)
            correct_val += (predicted_val == target_val_class).sum().item()

    val_acc = correct_val / total_val
    conf_matrix = confusion_matrix(all_targets, all_preds )

    return val_loss, val_acc, conf_matrix

def find_mode(results):
    if len(results) == 0:
        return -1
    return Counter(results).most_common(1)[0][0]

def preprocess_data(file_paths, img_roi, equal):
    def load_image(file_path):
        file_extension = os.path.splitext(file_path)[-1].lower()
        with open(file_path, 'rb') as image_file:
            if file_extension == '.yuv':
                image = np.frombuffer(image_file.read(128 * 128), dtype=np.uint8).reshape((128, 128))
                image = image.astype(np.float32) / 255.0
            elif file_extension == '.tiff':
                image = imageio.imread(image_file).astype(np.float32)
                if image.dtype == np.uint16:
                    image /= 65535
                else:
                    image /= 255
            else:
                raise ValueError("Unsupported file format")
        return image

    X_data = [load_image(file_path) for file_path in file_paths]
    X_data = np.array(X_data)
    X_processed = abs(X_data - np.roll(X_data, -1, axis=0))

    if equal:
        X_processed = exposure.equalize_hist(X_processed)

    X_processed = np.delete(X_processed, -1, axis=0)
    X_processed = X_processed[:, :img_roi, :img_roi]
    X_processed = np.expand_dims(X_processed, axis=1)
    X_processed = torch.tensor(X_processed, dtype=torch.float32)

    return X_processed

def model_inference(images, model, batch_size):
    model.eval()
    device = next(model.parameters()).device
    images = images.to(device)

    dataloader = DataLoader(TensorDataset(images), batch_size=batch_size, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for batch in dataloader:
            batch_images = batch[0]
            outputs = model(batch_images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())

    return all_preds

def evaluate_videos(structure, model, img_roi, num_classes, equal, img_frame, batch_size=32):
    all_preds = []
    all_targets = []
    all_videos = []

    for date_folder, subfolders_data in structure.items():
        for subfolder, video_folders in subfolders_data.items():
            for video_folder, segments in video_folders.items():
                segment_results = []

                for segment, segment_paths in segments.items():
                    if len(segment_paths) <= 1:
                        continue

                    limited_segment_paths = segment_paths[:img_frame]
                    images = preprocess_data(limited_segment_paths, img_roi, equal)
                    result = model_inference(images, model, batch_size)
                    segment_results.extend(result)

                overall_result = find_mode(segment_results)

                video_name = os.path.join(date_folder, subfolder, video_folder)

                if overall_result == -1:
                    print(f"Warning: No results for video {video_name}")
                    continue

                all_preds.append(overall_result)
                all_targets.append(int(subfolder))

                video_counter = Counter(segment_results)
                all_videos.append((video_name, video_counter))

    conf_matrix = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))

    return conf_matrix, all_videos

def calculate_cutoff_accuracy(conf_matrix):
    tp_top_left = np.sum(conf_matrix[0:2, 0:2])
    tp_bottom_right = np.sum(conf_matrix[2:, 2:])
    total_true_positives = tp_top_left + tp_bottom_right
    total_samples = np.sum(conf_matrix)
    accuracy = total_true_positives / total_samples if total_samples else 0
    return accuracy

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
    print("----------------Test----------------")
    date_folders = [d for d in os.listdir(args.data_path) if os.path.isdir(os.path.join(args.data_path, d))]
    print(date_folders)
    print("------------------------------------")
    print()

    kf = KFold(n_splits=len(date_folders))

    results = pd.DataFrame(columns=['Fold', 'Train Folders', 'Test Folder', 'Frame Confusion Matrix', 'Frame Accuracy', 'Frame F1 Score', 'Video Confusion Matrix', 'Video Accuracy', 'Video F1 Score', 'Cut-off 10^2 CFU/ml'])
    results_summary = {
        'Frame Accuracy': [],
        'Video Accuracy': [],
        'Cut-off 10^2 CFU/ml': [],
        'Frame F1 Score': [],
        'Video F1 Score': [],
    }

    cumulative_frame_conf_matrix = np.zeros((args.num_classes, args.num_classes))
    cumulative_video_conf_matrix = np.zeros((args.num_classes, args.num_classes))

    all_video_counters = []

    for fold_index, (train_index, val_index) in enumerate(kf.split(date_folders)):
        train_folders = [date_folders[i] for i in train_index]
        val_folder = [date_folders[val_index[0]]]

        print(f'Train: {train_folders}')
        print(f'Val: {val_folder}')

        test_structure = get_temporary_structure_with_stable_segments(
            args.data_path, val_folder, args.subfolders, args.stability_threshold, args.buffer_size)

        X_test, y_test = process_images(test_structure, args.img_roi, args.num_classes)
        test_loader = make_data_loader(X_test, y_test, args.batch_size)

        model = create_model(args.num_classes, args.model_type).to(device)
        model_path = os.path.join(args.models_dir, f"Fold_{fold_index + 1}_HTCNN.pth")
        model.load_state_dict(torch.load(model_path))

        frame_loss, frame_accuracy, frame_conf_matrix = evaluate_model(model, test_loader, device)
        frame_f1_scores = calculate_f1_scores(frame_conf_matrix)

        video_conf_matrix, video_counters = evaluate_videos(test_structure, model, args.img_roi, args.num_classes, args.equal, args.img_frame)
        video_f1_scores = calculate_f1_scores(video_conf_matrix)
        video_accuracy = np.trace(video_conf_matrix) / np.sum(video_conf_matrix) if np.sum(video_conf_matrix) else 0

        all_video_counters.extend(video_counters)

        print(f"Evaluating Fold {fold_index + 1}: test {val_folder}")
        print(f"Frame Confusion Matrix for Fold {fold_index + 1}:\n{frame_conf_matrix}")
        print(f"Frame Accuracy for Fold {fold_index + 1}: {frame_accuracy * 100:.2f}%\n")
        print(f"Video Confusion Matrix for Fold {fold_index + 1}:\n{video_conf_matrix}")
        print(f"Video Accuracy for Fold {fold_index + 1}: {video_accuracy * 100:.2f}%\n")

        if len(args.subfolders) == 5:
            accuracy_cutoff_revised = calculate_cutoff_accuracy(video_conf_matrix)
            print(f'Cut-off 10^2 CFU/ml Accuracy for Fold {fold_index + 1}: {accuracy_cutoff_revised * 100:.2f}%\n')

        train_folders_str = ', '.join(train_folders)
        test_folder_name = val_folder[0]

        fold_data = {
            'Fold': fold_index + 1,
            'Train Folders': train_folders_str,
            'Test Folder': test_folder_name,
            'Frame Confusion Matrix': frame_conf_matrix.tolist(),
            'Frame Accuracy': frame_accuracy,
            'Frame F1 Score': np.mean(frame_f1_scores),
            'Video Confusion Matrix': video_conf_matrix.tolist(),
            'Video Accuracy': video_accuracy,
            'Video F1 Score': np.mean(video_f1_scores)
        }
        if len(args.subfolders) == 5:
            fold_data['Cut-off 10^2 CFU/ml'] = accuracy_cutoff_revised

        fold_data_df = pd.DataFrame([fold_data])
        results = pd.concat([results, fold_data_df], ignore_index=True)
        results_summary['Frame Accuracy'].append(frame_accuracy)
        results_summary['Frame F1 Score'].append(np.mean(frame_f1_scores))
        results_summary['Video Accuracy'].append(video_accuracy)
        results_summary['Video F1 Score'].append(np.mean(video_f1_scores))

        if len(args.subfolders) == 5:
            results_summary['Cut-off 10^2 CFU/ml'].append(accuracy_cutoff_revised)

        cumulative_frame_conf_matrix += np.array(frame_conf_matrix)
        cumulative_video_conf_matrix += np.array(video_conf_matrix)

    total_frame_conf_matrix = cumulative_frame_conf_matrix
    total_video_conf_matrix = cumulative_video_conf_matrix

    total_frame_accuracy = sum(results_summary['Frame Accuracy']) / len(results_summary['Frame Accuracy'])
    total_video_accuracy = sum(results_summary['Video Accuracy']) / len(results_summary['Video Accuracy'])
    total_frame_f1 = sum(results_summary['Frame F1 Score']) / len(results_summary['Frame F1 Score'])
    total_video_f1 = sum(results_summary['Video F1 Score']) / len(results_summary['Video F1 Score'])
    total_cutoff_accuracy = sum(results_summary['Cut-off 10^2 CFU/ml']) / len(results_summary['Cut-off 10^2 CFU/ml']) if len(args.subfolders) == 5 else None

    print("Total Frame Confusion Matrix:\n", np.round(total_frame_conf_matrix).astype(int))
    print(f"Total Frame Accuracy: {total_frame_accuracy * 100:.2f}%")
    print(f"Total Frame F1 Score: {total_frame_f1 * 100:.2f}%")
    print("Total Video Confusion Matrix:\n", np.round(total_video_conf_matrix).astype(int))
    print(f"Total Video Accuracy: {total_video_accuracy * 100:.2f}%")
    print(f"Total Video F1 Score: {total_video_f1 * 100:.2f}%")
    if total_cutoff_accuracy is not None:
        print(f"Total Cut-off 10^2 CFU/ml Accuracy: {total_cutoff_accuracy * 100:.2f}%")

    total_results = pd.DataFrame({
        'Metric': ['Frame Accuracy', 'Video Accuracy', 'Frame F1 Score', 'Video F1 Score', 'Cut-off 10^2 CFU/ml Accuracy'],
        'Value': [
            f"{total_frame_accuracy * 100:.2f}%",
            f"{total_video_accuracy * 100:.2f}%",
            f"{total_frame_f1 * 100:.2f}%",
            f"{total_video_f1 * 100:.2f}%",
            f"{total_cutoff_accuracy * 100:.2f}%" if total_cutoff_accuracy is not None else "N/A"
        ]
    })

    excel_path = os.path.join(args.excel_dir, "experiment_results.xlsx")
    results.to_excel(excel_path, index=False)
    with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists="replace") as writer:
        total_results.to_excel(writer, sheet_name='Average Results', index=False)
        total_frame_conf_matrix_df = pd.DataFrame(total_frame_conf_matrix, columns=[f"Pred_{i}" for i in range(total_frame_conf_matrix.shape[1])])
        total_frame_conf_matrix_df.index = [f"True_{i}" for i in range(total_frame_conf_matrix_df.shape[0])]
        total_frame_conf_matrix_df.to_excel(writer, sheet_name='Avg Frame Confusion Matrix', index=True)
        total_video_conf_matrix_df = pd.DataFrame(total_video_conf_matrix, columns=[f"Pred_{i}" for i in range(total_video_conf_matrix.shape[1])])
        total_video_conf_matrix_df.index = [f"True_{i}" for i in range(total_video_conf_matrix_df.shape[0])]
        total_video_conf_matrix_df.to_excel(writer, sheet_name='Avg Video Confusion Matrix', index=True)

        # Add video counter information to Excel
        video_counter_df = pd.DataFrame(
            [(video, dict(counter)) for video, counter in all_video_counters],
            columns=['Video Name', 'Counter']
        )
        video_counter_df.to_excel(writer, sheet_name='Video Counters', index=False)

    print(f"Results and averages have been saved to {excel_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Configuration")
    parser.add_argument('--data_path', type=str, default='/home/jijang/projects/Bacteria/dataset/case_test/case16', help='Base path for dataset')
    parser.add_argument('--models_dir', type=str, default='/home/jijang/projects/Bacteria/models/case_test/240901_case16_torch_cnn_early_stop_best_kd', help='Directory to save models')
    parser.add_argument('--excel_dir', type=str, default='/home/jijang/projects/Bacteria/excel/case_test/240901_case16_torch_cnn_early_stop_best_kd', help='Directory to save excel results')
    parser.add_argument('--subfolders', nargs='+', default=['0', '1', '2', '3', '4'], help='List of subfolders for classes')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes')
    parser.add_argument('--img_frame', type=int, default=900, help='Number of image frames')
    parser.add_argument('--img_roi', type=int, default=128, help='Image region of interest size')
    parser.add_argument('--stability_threshold', type=int, default=350, help='Stability threshold for segment detection')
    parser.add_argument('--buffer_size', type=int, default=0, help='Buffer size around peaks to exclude')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--equal', action='store_true', help='Apply histogram equalization')
    parser.add_argument('--model_type', type=str, choices=['small', 'large'], default='small', help='Model type to use (small or large)')

    args = parser.parse_args()
    main(args)