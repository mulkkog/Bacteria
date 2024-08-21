import os

# 기본 경로 설정
base_path = '/home/jijang/projects/Bacteria/dataset/Razer'

# 클래스별 폴더 개수를 저장할 딕셔너리 초기화
total_counts = {str(i): 0 for i in range(5)}  # 0부터 4까지의 클래스를 가정
total_frame_counts = {str(i): 0 for i in range(5)}  # 프레임 수를 저장할 딕셔너리

# 기본 경로 아래의 모든 폴더를 탐색
for date_folder in os.listdir(base_path):
    date_path = os.path.join(base_path, date_folder)
    if os.path.isdir(date_path):  # 날짜 폴더인지 확인
        # 날짜별 클래스별 개수를 저장할 임시 딕셔너리
        date_counts = {str(i): 0 for i in range(5)}
        date_frame_counts = {str(i): 0 for i in range(5)}  # 날짜별 프레임 수를 저장할 임시 딕셔너리
        
        # 날짜 폴더 내의 모든 클래스 폴더를 탐색
        for class_folder in os.listdir(date_path):
            class_path = os.path.join(date_path, class_folder)

            if os.path.isdir(class_path):  # 클래스 폴더인지 확인
                # 클래스 폴더 내의 "비디오" 폴더(하위 폴더) 개수 카운트
                num_videos = 0
                num_frames = 0
                for video_folder in os.listdir(class_path):
                    video_path = os.path.join(class_path, video_folder)
                    if os.path.isdir(video_path):  # 비디오 폴더인지 확인
                        num_videos += 1
                        # 비디오 폴더 내의 프레임(.yuv 파일) 수 카운트
                        num_frames += len([f for f in os.listdir(video_path) if f.endswith('.yuv')])
                
                first_char = class_path.split('/')[-1].split('-')[0] if '-' in class_path.split('/')[-1] else class_path.split('/')[-1]
                if first_char.isdigit():
                    if first_char == '0':
                        class_folder = '0'
                    elif first_char == '1':
                        class_folder = '1'
                    elif first_char == '2':
                        class_folder = '2'
                    elif first_char == '3':
                        class_folder = '3'
                    elif first_char == '4':
                        class_folder = '4'

                date_counts[class_folder] += num_videos
                total_counts[class_folder] += num_videos
                date_frame_counts[class_folder] += num_frames
                total_frame_counts[class_folder] += num_frames
        
        # 날짜별 결과 출력
        print(f'{date_path}: ', end='')
        print(', '.join([f'{cls}: {count} (frames: {date_frame_counts[cls]})' for cls, count in date_counts.items()]))
        
# 전체 결과 출력
print('total: ', end='')
print(', '.join([f'{cls}: {count} (frames: {total_frame_counts[cls]})' for cls, count in total_counts.items()]))
