import os
import random
import argparse
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import KFold
from scipy.signal import find_peaks 
from skimage import exposure
from PIL import Image

# GPU 설정
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "7"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42) 

class ConvNet(nn.Module):
    def __init__(self, input_channels, num_classes): 
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool = nn.AdaptiveMaxPool2d((None, None))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global pooling to produce a fixed size output
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def create_model(num_classes):
    return ConvNet(1, num_classes)

 
# 이미지 파일 처리 함수
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



# 세그먼트 찾기 함수
def find_all_stable_segments(summed_diffs, peaks, buffer_size):
    total_frames = len(summed_diffs)
    excluded_indices = set()
    for peak in peaks:
        for i in range(max(0, peak - buffer_size), min(total_frames, peak + buffer_size + 1)):
            excluded_indices.add(i)
    valid_frames = [i for i in range(total_frames) if i not in excluded_indices]
    return valid_frames


# 임시 구조 생성 함수
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
            "input": self.X[idx].unsqueeze(1),
            "target": self.y[idx]
        }

def make_data_loader(X, y, batch_size, tag=''):
    dataset = torchLoader(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True if tag == 'train' else False, num_workers=4)
    return loader


def main(args):
    os.makedirs(args.models_dir, exist_ok=True)
    print("----------------Train----------------")

    # 날짜 폴더 찾기
    date_folders = [d for d in os.listdir(args.data_path) if os.path.isdir(os.path.join(args.data_path, d))]
    print(date_folders)
    print("-------------------------------------")

    # 전체 데이터셋의 임시 구조 생성
    entire_structure = get_temporary_structure_with_stable_segments(
        args.data_path, date_folders, args.subfolders, args.stability_threshold, args.buffer_size)

    kf = KFold(n_splits=len(date_folders))

    accuracy_results = []

    # 교차 검증
    for fold_index, (train_index, val_index) in enumerate(kf.split(date_folders)):
        train_folders = [date_folders[i] for i in train_index]
        val_folder = [date_folders[val_index[0]]]

        print(f'Train: {train_folders}')
        print(f'Val: {val_folder}')

        # Train 및 Validation 구조 생성
        train_structure = {folder: entire_structure[folder] for folder in train_folders if folder in entire_structure}
        val_structure = {folder: entire_structure[folder] for folder in val_folder if folder in entire_structure}
     
        X_train, y_train    = process_images(train_structure, args.img_roi, args.num_classes)
        X_val, y_val        = process_images(val_structure, args.img_roi, args.num_classes)

        # torch DataLoader : HxWx1 > 1xHxW
        train_loader = make_data_loader(X_train, y_train, args.batch_size, tag="train")
        val_loader = make_data_loader(X_val, y_val, args.batch_size)
        
        # torch model
        model = create_model(args.num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            pbar = tqdm(train_loader, leave=False, desc='train', smoothing=0.9)
            
            for batch in pbar:
                inp = batch["input"].to(device)
                target = batch["target"].to(device)
                output = model(inp)

                loss = criterion(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                _, target_class = torch.max(target.data, 1)
                total_train += target.size(0)
                correct_train += (predicted == target_class).sum().item()
            
            train_accuracy = 100 * correct_train / total_train

            # Validation during training
            model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_val  = batch["input"].to(device)
                    target_val = batch["target"].to(device)
                    output_val = model(input_val)

                    loss = criterion(output_val, target_val)
                    val_loss += loss.item()
                    _, predicted_val = torch.max(output_val.data, 1)
                    _, target_val_class = torch.max(target_val.data, 1)
                    total_val += target_val.size(0)
                    correct_val += (predicted_val == target_val_class).sum().item()

            val_accuracy = 100 * correct_val / total_val
            
            print(f'Epoch[{epoch}]-- Train Loss : {running_loss / len(train_loader):.5f}, Val Loss : {val_loss / len(val_loader):.5f}, Train Accuracy: {train_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%')
            model_save_path = os.path.join(args.models_dir, f"Fold_{fold_index + 1}_HTCNN.pth")
            torch.save(model.state_dict(), model_save_path)
            print()
            
        accuracy_results.append((train_accuracy, val_accuracy))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Configuration")
    parser.add_argument('--data_path',  type=str, default='/home/jijang/projects/Bacteria/dataset/case_test/case4', help='Base path for dataset')
    parser.add_argument('--models_dir', type=str, default='/home/jijang/projects/Bacteria/models/case_test/240820_case4_torch_vivit', help='Directory to save models')
    parser.add_argument('--subfolders', nargs='+', default=['0', '1', '2', '3'], help='List of subfolders for classes')   
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes')
    parser.add_argument('--img_frame', type=int, default=900, help='Number of image frames')
    parser.add_argument('--img_roi', type=int, default=96, help='Image region of interest size')
    parser.add_argument('--stability_threshold', type=int, default=350, help='Stability threshold for segment detection')
    parser.add_argument('--buffer_size', type=int, default=2, help='Buffer size around peaks to exclude')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--equal', action='store_false', help='Apply histogram equalization')

    args = parser.parse_args()
    main(args)