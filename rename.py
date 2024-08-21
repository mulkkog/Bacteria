import os
import re
import shutil

# 기본 디렉토리 설정
base_dir = "/home/jijang/projects/Bacteria/dataset/WaterP_TWT_forward/0527_29/0529-3"

# 숫자만 추출하는 함수
def extract_number(filename):
    numbers = re.findall(r'\d+', filename)
    return numbers[0] if numbers else None

# 파일명을 변경하는 함수
def rename_file(original_path, new_name):
    new_path = os.path.join(os.path.dirname(original_path), new_name)
    shutil.move(original_path, new_path)

# 숫자를 0으로 채워넣는 함수
def zero_pad_number(number, length=5):
    return str(number).zfill(length)

# 접두사 매핑 함수
def get_prefix(class_label, sub_label=""):
    prefix_mapping = {
        "0": "dw",
        "1": "e101",
        "2": "e102",
        "3": "e103",
        "4": "e104",
        "5": "e105"
    }
    prefix = prefix_mapping.get(class_label, "")
    if sub_label:
        prefix += f"-{sub_label}"
    return prefix

# 이미 처리된 파일/폴더인지 확인하는 함수
def is_already_processed(path):
    return any(prefix in path for prefix in ["dw", "e101", "e102", "e103", "e104", "e105"])

# 폴더와 파일을 재구성하는 메인 함수
def reorganize_folders(base_dir):
    for folder1 in os.listdir(base_dir):
        folder1_path = os.path.join(base_dir, folder1)
        class_label = folder1.split('-')[0]
        for folder2 in os.listdir(folder1_path):
            folder2_path = os.path.join(folder1_path, folder2)
            sub_label = folder1_path.split('-')[-1] if '-' in  os.path.basename(folder1_path) else ""
            prefix = get_prefix(class_label, sub_label)

            if is_already_processed(folder2_path):
                new_folder3_name = folder2
                new_folder3_path = os.path.join(folder1_path, new_folder3_name)
                pass  # 이미 처리된 폴더는 건너뜁니다.
            else:
                new_folder3_name = f"{prefix}_{folder2}_{os.path.basename(base_dir)}"
                new_folder3_path = os.path.join(folder1_path, new_folder3_name)
                shutil.move(folder2_path, new_folder3_path)

            # 파일 처리
            yuv_files = [f for f in os.listdir(new_folder3_path) if f.endswith('.yuv')]
            for yuv_file in yuv_files:
                number_only = extract_number(yuv_file)
                if len(number_only) == 5:
                    continue

                if number_only is not None:
                    original_file_path = os.path.join(new_folder3_path, yuv_file)
                    padded_number = zero_pad_number(number_only)
                    new_file_name = f"sensor_dump_{padded_number}.yuv"
                    rename_file(original_file_path, new_file_name)

            # 폴더명에 접두사 추가
            if not new_folder3_name.startswith(prefix):
                final_folder_name = f"{prefix}_{new_folder3_name}"
                final_folder_path = os.path.join(folder1_path, final_folder_name)
                shutil.move(new_folder3_path, final_folder_path)

# 메인 함수 실행
reorganize_folders(base_dir)
