import argparse
import random
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Lambda
from tensorflow.keras.models import Model
from sklearn.model_selection import KFold 
import gc
from tensorflow.keras import backend as K

# GPU 설정 및 메모리 성장 활성화
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Enabled memory growth for {len(gpus)} GPU(s).")
    except RuntimeError as e:
        print(e)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(42)

# --------------------------------------------------------------------
# 1) VAE 모델 정의
#    - Encoder: 입력 -> (Conv 블록 * N) -> Flatten -> Dense(mu), Dense(log_var)
#    - Sampling 레이어: reparameterization trick
#    - Decoder: 잠재 벡터 -> (Conv 블록 역순 * N) -> Output (원본과 동일 차원)
#    - VAE: 위의 구조를 통합하고, KL Divergence + Reconstruction Loss를 합산해 학습
# --------------------------------------------------------------------
def create_vae(input_shape, latent_dim=16):
    # -------------------------------------------------
    # Encoder
    # -------------------------------------------------
    encoder_inputs = Input(shape=input_shape, name="encoder_input")

    # 간단한 Conv 블록
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_inputs)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    # 잠재공간으로 펼치기
    x = Flatten()(x)

    # 잠재변수의 평균과 로그분산 추정
    mu = Dense(latent_dim, name="mu")(x)
    log_var = Dense(latent_dim, name="log_var")(x)

    # Reparameterization Trick
    def sampling(args):
        mu, log_var = args
        # 표준정규분포에서 샘플링한 noise
        epsilon = K.random_normal(shape=(K.shape(mu)[0], latent_dim))
        # z = mu + sigma * epsilon  (sigma = exp(log_var / 2))
        return mu + K.exp(0.5 * log_var) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([mu, log_var])

    encoder = Model(encoder_inputs, [mu, log_var, z], name='encoder')

    # -------------------------------------------------
    # Decoder
    # -------------------------------------------------
    decoder_inputs = Input(shape=(latent_dim,), name="z_input")
    # 이미지로 복원하기 위해 잠재벡터 -> FC -> Conv 차원으로 reshape
    # 인코더의 마지막 Conv Pool 결과가 어느 정도 크기인지 계산 필요
    # 예: img_roi=96, Conv 2번 Pooling -> 96/4=24, 채널 수는 64
    # 따라서 Flatten 이전의 shape가 (24, 24, 64) 라고 가정
    # Flatten 전 shape = 24 * 24 * 64 = 36864
    # 아래는 예시로 맞춰놓은 것이므로, 실제 사용 시 인코더 구조에 맞춰 수정 필요
    conv_dim = 24
    channel_dim = 64
    decoder_units = conv_dim * conv_dim * channel_dim  # 24*24*64 = 36864 (예시)

    x = Dense(decoder_units, activation='relu')(decoder_inputs)
    x = Reshape((conv_dim, conv_dim, channel_dim))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    decoder = Model(decoder_inputs, decoded, name='decoder')

    # -------------------------------------------------
    # VAE
    # -------------------------------------------------
    # encoder -> decoder
    mu, log_var, z = encoder(encoder_inputs)
    outputs = decoder(z)

    vae = Model(encoder_inputs, outputs, name='vae')

    # -------------------------------------------------
    # VAE Loss: Reconstruction Loss + KL Divergence
    # -------------------------------------------------
    # (1) 재구성 오류: MSE, BCE 등
    reconstruction_loss = tf.reduce_mean(
        tf.reduce_sum(
            tf.keras.losses.mean_squared_error(encoder_inputs, outputs),
            axis=[1,2]  # (height, width) 차원에 대해 합산
        )
    )

    # (2) KL Divergence
    # KLD = -0.5 * Σ(1 + log_var - mu^2 - exp(log_var))
    kl_loss = -0.5 * tf.reduce_mean(
        tf.reduce_sum(
            1 + log_var - tf.square(mu) - tf.exp(log_var),
            axis=1
        )
    )

    total_vae_loss = reconstruction_loss + kl_loss
    vae.add_loss(total_vae_loss)

    return encoder, decoder, vae

# --------------------------------------------------------------------
# 데이터셋 처리를 위한 유틸 함수들 (이전 AE 코드와 동일)
# --------------------------------------------------------------------
def get_all_structure(base_path, date_folders, subfolders):
    """
    주어진 폴더 구조에서, 모든 파일을 단일 세그먼트로 모으는 예시 함수.
    """
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
                            if video_files_sorted:
                                structure[date_folder][subfolder][video_name_folder] = {
                                    "segment_0": [os.path.join(video_folder_path, f) for f in video_files_sorted]
                                }
    return structure

def process_images(structure, img_roi, apply_equalization):
    """
    구조화된 파일 정보를 통해 이미지 불러오고, 전처리한 뒤,
    (N, img_roi, img_roi, 1) 텐서로 반환.
    """
    X_data = []

    for date_folder, subfolders in structure.items():
        for subfolder, video_folders in subfolders.items():
            for video_folder, segments in video_folders.items():
                for segment_id, file_paths in segments.items():
                    if len(file_paths) == 0:
                        continue
                    X = get_images(file_paths, apply_equalization)
                    X_data.append(X)

    if not X_data:
        raise ValueError("No image data found. Please check your dataset.")

    X_data = np.vstack(X_data).astype(np.float32)
    # 채널 차원 추가: (N, H, W, 1)
    X_data = np.expand_dims(X_data, axis=-1)
    # ROI에 맞춰 자르기
    X_data = X_data[:, :img_roi, :img_roi, :]

    return X_data

def get_images(file_paths, apply_equalization):
    X = []
    for file_path in file_paths:
        if file_path.endswith('.tiff'):
            image = load_tiff_image(file_path, apply_equalization)
        else:
            image = load_yuv_image(file_path, apply_equalization)
        X.append(image)
    return np.array(X)

def load_tiff_image(file_path, apply_equalization):
    import imageio
    from skimage import exposure

    image = imageio.imread(file_path)
    # 16-bit or 8-bit check
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
    from skimage import exposure

    try:
        file_size = os.path.getsize(file_path)
        required_size = 128 * 128

        if file_size < required_size:
            raise ValueError(f"File '{file_path}' size ({file_size} bytes) is smaller than required size ({required_size} bytes).")

        with open(file_path, 'rb') as file:
            raw_data = file.read(required_size)

        if len(raw_data) != required_size:
            raise ValueError(f"Read data size ({len(raw_data)} bytes) from file '{file_path}' "
                             f"does not match required size ({required_size} bytes).")

        image = np.frombuffer(raw_data, dtype=np.uint8).reshape((128, 128))
        image = image.astype(np.float32) / 255.0

        if apply_equalization:
            image = exposure.equalize_hist(image)

        return image

    except Exception as e:
        print(f"Error loading YUV image from file '{file_path}': {e}")
        raise

# --------------------------------------------------------------------
# 메인 실행 함수
# --------------------------------------------------------------------
def main(args):
    os.makedirs(args.models_dir, exist_ok=True)
    excel_path = os.path.join(args.models_dir, "test_results.xlsx")
    df_dummy = pd.DataFrame()
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_dummy.to_excel(writer)

    date_folders = [d for d in os.listdir(args.base_path) if os.path.isdir(os.path.join(args.base_path, d))]
    kf = KFold(n_splits=len(date_folders))
    reconstruction_errors = []

    for fold_index, (train_index, val_index) in enumerate(kf.split(date_folders)):
        print(f"Starting Fold {fold_index + 1}")
        train_folders = [date_folders[i] for i in train_index]
        val_folder = [date_folders[val_index[0]]]
        
        # 전체 파일을 단일 세그먼트로 받아오는 예시 구조
        train_structure = get_all_structure(args.base_path, train_folders, args.subfolders)
        val_structure = get_all_structure(args.base_path, val_folder, args.subfolders)
        
        # 이미지 로드/전처리
        X_train = process_images(train_structure, args.img_roi, args.equalize)
        X_val = process_images(val_structure, args.img_roi, args.equalize)
        
        # -------------------------------------------------
        # VAE 생성
        # -------------------------------------------------
        encoder, decoder, vae = create_vae((args.img_roi, args.img_roi, 1), latent_dim=args.latent_dim)

        # 컴파일 (커스텀 loss가 vae.add_loss로 등록되어 있으므로, 따로 loss='...' 지정 X)
        optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)
        vae.compile(optimizer=optimizer)
        vae.summary()

        # 학습
        vae.fit(X_train, 
                epochs=args.epochs, 
                batch_size=64, 
                validation_data=(X_val, None),
                verbose=1)

        # 모델 저장
        model_save_path = os.path.join(args.models_dir, f"Fold_{fold_index + 1}_vae.h5")
        vae.save(model_save_path)
        
        # Reload the model
        del vae
        gc.collect()
        vae = tf.keras.models.load_model(model_save_path, compile=False)  # compile=False 필수 (custom loss)
        
        # Validation reconstruction
        reconstructed = vae.predict(X_val)
        mse = np.mean(np.power(X_val - reconstructed, 2), axis=(1,2,3))
        reconstruction_errors.append(mse)
        
        # Fold 결과 엑셀 저장
        fold_result = pd.DataFrame({
            'Fold': [fold_index + 1],
            'Mean Reconstruction Error': [np.mean(mse)],
            'Std Reconstruction Error': [np.std(mse)]
        })
        with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a') as writer:
            fold_result.to_excel(writer, sheet_name=f'Fold_{fold_index + 1}', index=False)

        # TFLite 변환 (Sampling 레이어 등으로 인해 변환 시 이슈가 있을 수 있으므로 참고)
        converter = tf.lite.TFLiteConverter.from_keras_model(vae)
        tflite_model = converter.convert()
        tflite_save_path = os.path.join(args.models_dir, f"Fold_{fold_index + 1}_vae.tflite")
        with open(tflite_save_path, 'wb') as f:
            f.write(tflite_model)

        # 메모리 정리
        K.clear_session()
        del vae
        gc.collect()
        print(f"Completed Fold {fold_index + 1} and cleared memory.\n")

    # 전체 폴드 결과 정리
    all_reconstruction_errors = np.concatenate(reconstruction_errors)
    final_results = pd.DataFrame({
        'Fold': range(1, len(reconstruction_errors) + 1),
        'Mean Reconstruction Error': [np.mean(err) for err in reconstruction_errors],
        'Std Reconstruction Error': [np.std(err) for err in reconstruction_errors]
    })
    mean_rec_error = np.mean([np.mean(err) for err in reconstruction_errors])
    std_rec_error = np.std([np.mean(err) for err in reconstruction_errors])
    final_summary = pd.DataFrame({
        'Metric': ['Mean Reconstruction Error', 'Standard Deviation'],
        'Value': [mean_rec_error, std_rec_error]
    })
    
    with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a') as writer:
        final_results.to_excel(writer, sheet_name='Individual Folds', index=False)
        final_summary.to_excel(writer, sheet_name='Summary', index=False)
    
    print("Cross Validation Complete. Reconstruction Errors saved to Excel.")

# --------------------------------------------------------------------
# 시각화 함수 (선택사항)
# --------------------------------------------------------------------
def visualize_reconstructions(X, reconstructed, fold_index, save_dir, num_samples=5):
    import matplotlib.pyplot as plt
    os.makedirs(os.path.join(save_dir, f'Fold_{fold_index + 1}_Reconstructions'), exist_ok=True)
    for i in range(num_samples):
        plt.figure(figsize=(6,3))
        # Original
        plt.subplot(1,2,1)
        plt.imshow(X[i].squeeze(), cmap='gray')
        plt.title('Original')
        plt.axis('off')
        # Reconstructed
        plt.subplot(1,2,2)
        plt.imshow(reconstructed[i].squeeze(), cmap='gray')
        plt.title('Reconstructed')
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f'Fold_{fold_index + 1}_Reconstructions/sample_{i+1}.png'))
        plt.close()

# --------------------------------------------------------------------
# 파라미터 파싱
# --------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="Script to train and evaluate a Variational Autoencoder (VAE).")
    parser.add_argument('--base_path', type=str, default='/home/jijang/ssd_data/projects/Bacteria/dataset/stable', help='Base directory for the dataset.')
    parser.add_argument('--models_dir', type=str, default='/home/jijang/ssd_data/projects/Bacteria/models/stable_recon_vae', help='Directory where models are saved.')
    parser.add_argument('--subfolders', type=str, nargs='+', default=['0', '1'], help='Subfolders to include as classes.')
    parser.add_argument('--img_roi', type=int, default=96, help='Region of interest size for each image.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model.')
    parser.add_argument('--latent_dim', type=int, default=16, help='Dimension of the latent space.')
    parser.add_argument('--equalize', action='store_true', help='Apply histogram equalization to images.')
    return parser.parse_args()

# --------------------------------------------------------------------
# 실행 구문
# --------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_arguments()
    main(args)
