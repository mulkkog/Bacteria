import argparse
import random
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import pandas as pd
from tensorflow.keras import layers, models, losses, optimizers, datasets

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from scipy.signal import find_peaks
from PIL import Image
import tensorflow_model_optimization as tfmot

# GPU 설정
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "6"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(42)

def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (5, 5), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

def create_conv_small_net(input_shape, num_classes):
    model = Sequential([  
        Conv2D(8, kernel_size=(3, 3), strides=(3, 3), activation='relu', padding='valid', input_shape=input_shape), 
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'), 
        Conv2D(16, kernel_size=(2, 2), strides=(2, 2), activation='relu', padding='valid'), 
        MaxPooling2D((2, 2)),
        Flatten(), 
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')  # 다중 클래스 분류를 위해 softmax 사용
    ])
    
    return model

def process_images(structure, img_roi, num_classes):
    X_data, y_data = [], []

    for date_folder, subfolders in structure.items():
        for subfolder, video_folders in subfolders.items():
            for video_folder, segments in video_folders.items():
                for segment_id, file_paths in segments.items():
                    if len(file_paths) <= 1:
                        continue

                    X = get_images(file_paths)

                    for _ in range(len(file_paths) - 1):
                        y_data.append(subfolder)
                    
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
    return [i for i in range(total_frames) if i not in excluded_indices]

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
    return images

def process_video_files(folder_path, file_names, stability_threshold, buffer_size):
    if not file_names:
        print("The folder is empty. No files to process.")
        return []
    if file_names[0].endswith('.yuv'):
        images = load_yuv_images(folder_path, file_names)
    elif file_names[0].endswith('.tiff'):
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

def representative_dataset_gen(X_train):
    for input_value in tf.data.Dataset.from_tensor_slices(X_train).batch(1).take(100):
        yield [input_value]

def apply_qat(model):
    # 양자화 인식 훈련(QAT) 적용
    quantize_model = tfmot.quantization.keras.quantize_model
    qat_model = quantize_model(model)
    
    # 컴파일
    qat_model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    
    return qat_model
 
def distillation_loss(y_true, y_pred, teacher_pred, temperature=3, alpha=0.1):
    soft_teacher_pred = tf.nn.softmax(teacher_pred / temperature)
    soft_student_pred = tf.nn.softmax(y_pred / temperature)
    
    # Knowledge Distillation 손실 (KLD)
    distillation_loss = losses.KLDivergence()(soft_teacher_pred, soft_student_pred)
    
    # Cross-Entropy 손실
    hard_loss = losses.sparse_categorical_crossentropy(y_true, y_pred)
    
    # 두 손실의 조합
    return alpha * distillation_loss + (1 - alpha) * hard_loss

class DistillationLoss(losses.Loss):
    def __init__(self, teacher_predictions, temperature=3, alpha=0.1, name='distillation_loss'):
        super(DistillationLoss, self).__init__(name=name)
        self.teacher_predictions = teacher_predictions
        self.temperature = temperature
        self.alpha = alpha

    def call(self, y_true, y_pred):
        # 현재 배치 크기 가져오기
        batch_size = tf.shape(y_pred)[0]
        
        # teacher_predictions를 배치 크기에 맞게 슬라이싱
        teacher_pred_batch = tf.slice(self.teacher_predictions, [0, 0], [batch_size, -1])

        # 온도를 적용한 softmax 계산
        soft_teacher_pred = tf.nn.softmax(teacher_pred_batch / self.temperature)
        soft_student_pred = tf.nn.softmax(y_pred / self.temperature)

        # distillation loss 계산 (Kullback-Leibler Divergence)
        distillation_loss = losses.KLDivergence()(soft_teacher_pred, soft_student_pred)

        # Cross-Entropy 손실 계산
        hard_loss = losses.categorical_crossentropy(y_true, y_pred)

        # 두 손실을 결합하여 반환
        return self.alpha * distillation_loss + (1 - self.alpha) * hard_loss


def main(args):
    os.makedirs(args.student_model_dir, exist_ok=True)
    excel_path = os.path.join(args.student_model_dir, "test_results.xlsx")
    df_dummy = pd.DataFrame()
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_dummy.to_excel(writer)

    date_folders = sorted([d for d in os.listdir(args.base_path) if os.path.isdir(os.path.join(args.base_path, d))])
    print(f"Total date folders found: {date_folders}")
    kf = KFold(n_splits=len(date_folders))
    accuracy_results = []

    for fold_index, (train_index, val_index) in enumerate(kf.split(date_folders)):
        teacher_model_path = os.path.join(args.teacher_model_dir, f"Fold_{fold_index + 1}_model.h5")
        
        train_folders = [date_folders[i] for i in train_index]
        val_folder = [date_folders[val_index[0]]]
        train_structure = get_temporary_structure_with_stable_segments(args.base_path, train_folders, args.subfolders, args.stability_threshold, args.buffer_size)
        val_structure = get_temporary_structure_with_stable_segments(args.base_path, val_folder, args.subfolders, args.stability_threshold, args.buffer_size)
        X_train, y_train = process_images(train_structure, args.img_roi, args.num_classes)
        X_val, y_val = process_images(val_structure, args.img_roi, args.num_classes)
        
        with tfmot.quantization.keras.quantize_scope():
            teacher_model = load_model(teacher_model_path)
            teacher_model.trainable = False

        teacher_predictions = teacher_model.predict(X_train)

        student_model = create_conv_small_net((args.img_roi, args.img_roi, 1), args.num_classes)
        student_model = apply_qat(student_model)
        
        # Compile the student model with a custom loss (knowledge distillation)
        student_model.compile(optimizer='adam', loss=DistillationLoss(teacher_predictions), metrics=['accuracy'])

        # Early Stopping 및 ModelCheckpoint 콜백 설정
        callbacks = []
        if args.early_stop:
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=args.save_best)
            callbacks.append(early_stopping)

        if args.save_best:
            checkpoint_path = os.path.join(args.student_model_dir, f"Fold_{fold_index + 1}_qat_model.h5")
            model_checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1)
            callbacks.append(model_checkpoint)

        # QAT 모델 훈련 
        student_model.fit(X_train, y_train, epochs=args.epochs, verbose=1, batch_size=64, validation_data=(X_val, y_val), callbacks=callbacks)
        
        # 모델 저장 (save_best 옵션이 활성화되지 않은 경우 마지막 모델 저장)
        if not args.save_best:
            model_save_path = os.path.join(args.student_model_dir, f"Fold_{fold_index + 1}_qat_model.h5")
            student_model.save(model_save_path)
        else:
            # EarlyStopping 사용 시 restore_best_weights=True인 경우 이미 best 모델이 로드된 상태
            model_save_path = checkpoint_path
            print(f"Fold {fold_index + 1}: Best model saved at {model_save_path}")

        # Save the model in SavedModel format
        saved_model_path = os.path.join(args.student_model_dir, f"Fold_{fold_index + 1}_qat_saved_model")
        student_model.save(saved_model_path, save_format='tf')
        print(f"Fold {fold_index + 1}: Model also saved in SavedModel format at {saved_model_path}")

        # Evaluate the QAT model
        val_loss, val_acc = student_model.evaluate(X_val, y_val)
        accuracy_results.append(val_acc)
        fold_result = pd.DataFrame({'Fold': [fold_index + 1], 'Accuracy': [val_acc]})
        with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a') as writer:
            fold_result.to_excel(writer, sheet_name=f'Fold_{fold_index + 1}', index=False)
        
        # Generate confusion matrix
        y_pred = student_model.predict(X_val)
        y_pred = np.argmax(y_pred, axis=1)
        y_test = np.argmax(y_val, axis=1)
        result = confusion_matrix(y_test, y_pred)
        print(result)

        # TFLite 모델 변환: 선택한 양자화 방식에 따라 INT8 또는 FP16 적용
        converter = tf.lite.TFLiteConverter.from_keras_model(student_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if args.quantization_type.lower() == "int8":
            converter.representative_dataset = lambda: representative_dataset_gen(X_train)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        elif args.quantization_type.lower() == "fp16":
            converter.target_spec.supported_types = [tf.float16]
        else:
            raise ValueError("Unsupported quantization type: " + args.quantization_type)
        tflite_model = converter.convert()

        # Save the TFLite model (파일명에 양자화 방식을 포함)
        tflite_save_path = os.path.join(args.student_model_dir, f"Fold_{fold_index + 1}_qat_model_{args.quantization_type}.tflite")
        with open(tflite_save_path, 'wb') as f:
            f.write(tflite_model)

    # Final results
    final_results = pd.DataFrame({'Fold': range(1, len(accuracy_results) + 1), 'Accuracy': accuracy_results})
    mean_accuracy = np.mean(accuracy_results)
    std_accuracy = np.std(accuracy_results)
    final_summary = pd.DataFrame({'Metric': ['Mean Accuracy', 'Standard Deviation'], 'Value': [mean_accuracy, std_accuracy]})
    with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a') as writer:
        final_results.to_excel(writer, sheet_name='Individual Folds', index=False)
        final_summary.to_excel(writer, sheet_name='Summary', index=False)
    
    print("Cross Validation Complete. Results saved to Excel.")



def parse_arguments():
    parser = argparse.ArgumentParser(description="Script to train and evaluate a quantization-aware CNN model on image data.")
    parser.add_argument('--base_path', type=str, default='/home/jijang/ssd_data/projects/Bacteria/dataset/case_test/case16', help='Base directory for the dataset.')
    parser.add_argument('--teacher_model_dir', type=str, default='/home/jijang/ssd_data/projects/Bacteria/models/case_test/2025_case16', help='Directory where teacher model weights are stored.')
    parser.add_argument('--student_model_dir', type=str, default='/home/jijang/ssd_data/projects/Bacteria/models/case_test/2025_case16_qat_int8', help='Directory where student models are saved.')
    parser.add_argument('--subfolders', type=str, nargs='+', default=['0', '1', '2', '3', '4'], help='Subfolders to include as classes.')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes to predict.')
    parser.add_argument('--img_frame', type=int, default=900, help='Frame size of the images.')
    parser.add_argument('--img_roi', type=int, default=96, help='Region of interest size for each image.')
    parser.add_argument('--stability_threshold', type=int, default=350, help='Threshold for stability in video segmentation.')
    parser.add_argument('--buffer_size', type=int, default=2, help='Buffer size around detected peaks in stability analysis.')
    parser.add_argument('--learning_rate', type=float, default=0.00001, help='Learning rate for the optimizer.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model.')
    parser.add_argument('--early_stop', action='store_true', help='Enable early stopping during training.')
    parser.add_argument('--save_best', action='store_true', help='Save only the best model based on validation accuracy.')
    parser.add_argument('--quantization_type', type=str, default='int8', choices=['int8', 'fp16'], help='Type of quantization to apply during TFLite conversion (int8 or fp16).')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
 