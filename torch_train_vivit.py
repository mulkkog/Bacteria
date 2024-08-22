import os
import random
import argparse
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import KFold
from scipy.signal import find_peaks
from skimage import exposure
from PIL import Image
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from module import Attention, PreNorm, FeedForward

# GPU 설정
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

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

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class ViViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, num_frames, dim=192, depth=4, heads=3, pool='cls', in_channels=1, dim_head=64, dropout=0.,
                 emb_dropout=0., scale_dim=4, ):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches + 1, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = self.to_patch_embedding(x.unsqueeze(1))
        b, t, n, _ = x.shape

        cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b=b, t=t)
        x = torch.cat((cls_space_tokens, x), dim=2)
        x += self.pos_embedding[:, :, :(n + 1)]
        x = self.dropout(x)

        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x)
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)

        cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        x = self.temporal_transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        return self.mlp_head(x)


def create_model(img_roi, patch_size, num_classes, img_frame):
    return ViViT(img_roi, patch_size, num_classes, img_frame)


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


# TIFF 이미지 로드 함수
def load_tiff_images(folder_path, file_names):
    images = []
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        image = Image.open(file_path)

        # Convert image to numpy array
        image = np.array(image)

        # Check the data type before conversion
        if image.dtype == np.uint16:
            image = image.astype(np.float32) / 65535  # Normalizing 16-bit image
        else:
            image = image.astype(np.float32) / 255  # Normalizing 8-bit image

        images.append(image)
    return np.array(images)


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
def process_video_files(folder_path, file_names, stability_threshold, buffer_size, equal=True):
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

    if equal:
        image_diffs = exposure.equalize_hist(image_diffs)

    summed_diffs = image_diffs.sum(axis=(1, 2))

    # Find peaks in the 1D array of summed differences
    peaks, _ = find_peaks(summed_diffs, height=stability_threshold)

    # Find top stable segments
    stable_segments = find_all_stable_segments(summed_diffs, peaks, buffer_size)
    return stable_segments


# 이미지 가져오기 함수
def get_images(file_paths):
    images = []
    for file_path in file_paths:
        with open(file_path, 'rb') as image_file:
            image = np.frombuffer(image_file.read(128 * 128), dtype=np.uint8).reshape((128, 128))
            image = np.array(image) / 255.0
            images.append(image)
    images = np.array(images)  # 리스트를 numpy ndarray로 변환
    return torch.tensor(images)  # numpy ndarray를 torch tensor로 변환



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


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.epochs_no_improve = 0
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.epochs_no_improve = 0
        if self.early_stop and self.verbose:
            print("조기 종료: 검증 손실이 개선되지 않았습니다.")

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
        model = create_model(args.img_roi, args.patch_size, args.num_classes, 1).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = nn.CrossEntropyLoss()

        early_stopper = EarlyStopping(patience=5, verbose=True) if args.early_stop else None

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
                    input_val = batch["input"].to(device)
                    target_val = batch["target"].to(device)
                    output_val = model(input_val)

                    loss = criterion(output_val, target_val)
                    val_loss += loss.item()
                    _, predicted_val = torch.max(output_val.data, 1)
                    _, target_val_class = torch.max(target_val.data, 1)
                    total_val += target_val.size(0)
                    correct_val += (predicted_val == target_val_class).sum().item()

            if early_stopper:
                early_stopper(val_loss / len(val_loader))
                if early_stopper.early_stop:
                    print(f"Epoch {epoch}: 조기 종료됨")
                    break

            val_accuracy = 100 * correct_val / total_val

            print(f'Epoch[{epoch}]-- Train Loss : {running_loss / len(train_loader):.5f}, Val Loss : {val_loss / len(val_loader):.5f}, Train Accuracy: {train_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%')
            model_save_path = os.path.join(args.models_dir, f"Fold_{fold_index + 1}_HTCNN.pth")
            torch.save(model.state_dict(), model_save_path)
            print()

        accuracy_results.append((train_accuracy, val_accuracy))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Configuration")
    parser.add_argument('--data_path',  type=str, default='/home/jijang/projects/Bacteria/dataset/case_test/case16', help='Base path for dataset')
    parser.add_argument('--models_dir', type=str, default='/home/jijang/projects/Bacteria/models/case_test/240822_case16_torch_vivit_early_stop', help='Directory to save models')
    parser.add_argument('--subfolders', nargs='+', default=['0', '1', '2', '3'], help='List of subfolders for classes')   
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes')
    parser.add_argument('--img_frame', type=int, default=300, help='Number of image frames')
    parser.add_argument('--img_roi', type=int, default=96, help='Image region of interest size')
    parser.add_argument('--stability_threshold', type=int, default=350, help='Stability threshold for segment detection')
    parser.add_argument('--buffer_size', type=int, default=2, help='Buffer size around peaks to exclude')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size') 
    parser.add_argument('--patch_size', type=int, default=16, help='Patch size')
    parser.add_argument('--early_stop', action='store_false', help='Enable early stopping based on validation loss')
    
    args = parser.parse_args()
    main(args)
