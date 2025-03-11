# Bacteria Classification

이 프로젝트는 **이미지**를 입력으로 받아, 프레임 간 차분(Temporal Difference)을 활용해 CNN을 학습/검증하는 파이프라인입니다. 또한 **K-Fold Cross Validation**을 적용하여 폴더 단위로 훈련 세트와 검증 세트를 분리합니다.

---

## 전체 흐름 요약

1. **데이터셋 디렉토리 구성**  
   - `base_path` 내부에 날짜별 폴더(예: `20231201`, `20231202` 등) 존재  
   - 각 날짜 폴더는 `subfolder`(클래스) 폴더를 포함  
   - 각 `subfolder` 내에서 여러 영상 폴더(`video_name_folder`)가 존재하며, 이 폴더에는 YUV(또는 TIFF) 파일들이 순서대로 저장

2. **K-Fold Cross Validation**  
   - 날짜 폴더 수만큼의 Fold를 구성  
   - 각 Fold에서 **1개 폴더**만 검증(Validation) 폴더로, 나머지 폴더들은 학습(Training) 폴더로 사용  
   - Fold마다 안정 구간을 추출한 뒤(`get_temporary_structure_with_stable_segments`), CNN 학습 → 평가 → 결과/혼동행렬/모델(TFLite 변환 포함) 저장

3. **안정 구간(Stable segments) 추출**  
   - 영상 폴더 내 **연속된 프레임(YUV 파일들)을 모두 로드**  
   - 프레임 간 차분(diff)을 합산하여 **움직임이 큰 지점(피크)**을 `find_peaks`로 찾음  
   - 피크 주변(± buffer_size) 프레임들을 제외한 **나머지 구간**이 "안정 구간"이 됨  
   - 안정 구간을 연속된 프레임 덩어리(segment)로 묶어, 이후 학습용 데이터로 사용

4. **YUV 파일 로드**  
   - 단일 YUV 파일(128×128, 8-bit 단일 채널) → `np.uint8` 형태로 읽어온 뒤 `(128, 128)`로 Reshape  
   - `0~255` 범위를 `0~1`로 정규화  
   - (옵션) 히스토그램 평활화(`exposure.equalize_hist`) 적용

5. **차분을 통한 학습용 X, y 생성**  
   - 하나의 안정 구간에 프레임이 예: 5장이라면 → 4개의 차분(Temporal difference) 이미지를 얻게 됨  
   - **각 차분 결과**가 곧 **CNN 입력 샘플**  
   - 클래스 라벨은 `subfolder` 이름(예: "0", "1" 등)을 통해 One-hot 형태로 변환
   - ROI 크기(`img_roi`, 기본 96×96)만큼 잘라 최종 **X** 형성  
   - 이에 대응하는 **y** 라벨과 함께 학습용 세트 구축

6. **CNN 모델 학습 및 평가**  
   - 간단한 CNN 구조(Conv2D, MaxPooling2D, Flatten, Dense 등)를 사용  
   - Adam 옵티마이저로 학습 후, 검증 세트로 정확도/혼동행렬 확인  
   - 학습 완료된 모델(`.h5`)과 TFLite(`.tflite`) 모델을 저장하고 결과를 엑셀(`test_results.xlsx`)에 기록

---

## 주요 실행 파일 및 함수

- **`main(args)`**  
  전체 K-Fold 학습/평가 프로세스를 관장하며, 학습 결과(혼동 행렬, 정확도, 모델 파일 등)를 저장합니다.

- **`get_temporary_structure_with_stable_segments(...)`**  
  날짜/클래스/영상 폴더에 대해 **안정 구간(Stable segments)** 을 찾고, 각 구간별 프레임 파일 경로를 구조체로 묶어 반환합니다.

- **`process_images(...)`**  
  위에서 만든 구조체를 순회하며 **YUV(또는 TIFF) 파일을 실제로 읽고**, 프레임 간 차분 기반의 CNN 입력(`X`)과 라벨(`y`)을 생성합니다.

- **`load_yuv_image(...)`**  
  단일 YUV 파일(128×128, 8-bit)을 0~1 범위의 `float32` 배열로 로딩. (필요 시 히스토그램 평활화 가능)

- **`create_model(...)`**  
  간단한 CNN 모델(Conv2D → MaxPooling → Dense) 생성.

---

## 사용 예시

```bash
python train_script.py \
  --base_path /path/to/base/dataset \
  --models_dir /path/to/save/models \
  --subfolders 0 1 2 3 \
  --num_classes 4 \
  --img_roi 96 \
  --stability_threshold 350 \
  --buffer_size 2 \
  --learning_rate 0.0001 \
  --epochs 200 \
  --equalize
