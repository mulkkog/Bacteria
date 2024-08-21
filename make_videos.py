import os
import glob

# 기본 경로 설정
base_input_path = "/home/jijang/projects/Bacteria/dataset/E_coli"
base_output_path = "/home/jijang/projects/Bacteria/video/E_coli"

# Bash 스크립트 시작
bash_script = '''#!/bin/bash

'''

dates = os.listdir(base_input_path)

# 각 디렉토리에 대해 처리 
for day in dates:
    b_class = os.listdir(os.path.join(base_input_path, day))

    for cls in b_class:
        input_dir = os.path.join(base_input_path, day, cls)
        videos = os.listdir(input_dir)
        for video in videos:
            output_dir = os.path.join(base_output_path, day, cls, video)
            # 폴더 생성
            os.makedirs(output_dir, exist_ok=True)
            
            # YUV 파일 처리
            input_files_yuv = sorted(glob.glob(os.path.join(input_dir, video, "*.yuv")))
            if input_files_yuv:
                output_file_mp4 = os.path.join(output_dir, video + '.mp4')
                bash_script += (
                    'ffmpeg -f rawvideo -r 5 -pix_fmt gray -s:v 128x128 -i <(cat {}) -c:v libx265 '
                    '-pix_fmt yuv420p -x265-params lossless=1 "{}"\n'.format(
                        " ".join(f'"{f}"' for f in input_files_yuv), output_file_mp4
                    )
                )

# 스크립트 파일 작성 및 실행 권한 설정
script_filename = 'convert_videos.sh'
with open(script_filename, 'w') as file:
    file.write(bash_script)

os.chmod(script_filename, 0o755)
