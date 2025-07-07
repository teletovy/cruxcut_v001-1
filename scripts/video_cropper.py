import cv2
import torch
import os
import matplotlib.pyplot as plt
from moviepy.editor import *
import numpy as np
import mediapipe as mp
import time
from ultralytics import YOLO
import argparse

global w_margin, h_margin
global w_margin_list, h_margin_list
global margin_ratio
global square_mode

sampling_rate_default = 30
sampling_rate_tight = 5
w_margin_list = []
h_margin_list = []
margin_ratio = 1.6
square_mode = True

def yolov8(frame, model, pre_center_x, pre_center_y, show=False, save=False):
    # 대상 객체를 탐지했는지 여부를 나타내는 boolean 변수
    detected = False
    
    w = frame.shape[1]
    w_start = int(w // 4)
    w_end = int(w * 3 // 4)
    w_mid = w // 2
    # slice_frame = frame[:, w_start:w_end]
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame, stream=True, verbose=False)

    center_x, center_y = -1, -1
    
    frame_center_x, frame_center_y = frame.shape[1] // 2, frame.shape[0] // 2
    center_x_list = []
    center_y_list = []
    distance_list = []

    most_closed_center_x = 10000   

    for result in results:
        for box in result.boxes:
            pos = box.xyxy[0].tolist()
            pos = [round(x) for x in pos]
            x1, y1, x2, y2 = pos
            conf = box.conf[0].item()
            class_id = int(box.cls[0].item())
            class_str = result.names[class_id]
            
            if class_id == 0:
                object_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                dist_from_pre_center = np.sqrt((pre_center_x - object_center[0])**2 + (pre_center_y - object_center[1])**2)
                dist_from_center = np.abs(object_center[0] - frame_center_x)
                
                if object_center[0] < w_start or object_center[0] > w_end:
                    continue
                
                # if dist_from_pre_center > w//4:
                #     continue
                
                if dist_from_center < most_closed_center_x:
                    if detected:
                        print(detected)
                    most_closed_center_x = dist_from_center
                    center_x = object_center[0]
                    center_y = object_center[1]
                    
                    w_margin = (x2 - x1)* margin_ratio //2
                    h_margin = (y2 - y1)* margin_ratio //2

                    w_margin_list.append(w_margin)
                    h_margin_list.append(h_margin)
                
                    # for saving frame     
                    if save:
                        save = False
                        save_dir = os.path.join(os.getcwd(), 'detection_success')
                        idx = 0
                        image_name = f'img_{idx}.png'
                        while os.path.exists(os.path.join(save_dir, image_name)):
                            idx += 1
                            image_name = f'img_{idx}.png'
                        image_path = os.path.join(save_dir, image_name)
                        result_img = results.render()[0]
                        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                        cv2.imwrite(image_path, result_img)

    # 프레임에서 객체를 찾지 못한 경우 이전 프레임의 중심점을 사용  
    if center_x == -1 or center_y == -1:
        center_x, center_y = pre_center_x, pre_center_y
        
        # for data acquisition
        path = os.path.join(os.getcwd(), 'detection_fail')
        idx = 0
        image_name = f'img_{idx}.png'
        while os.path.exists(os.path.join(path, image_name)):
            idx += 1
            image_name = f'img_{idx}.png'
        image_path = os.path.join(path, image_name)
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_path, frame)
        
    return center_x, center_y


# def yolov8(frame, model, pre_center_x, pre_center_y, show=False, save=False):
    
#     detected = False
#     global w_margin_list, h_margin_list  

#     w = frame.shape[1]
#     w_start = int(w // 4)
#     w_end = int(w * 3 // 4)
    
#     results = model(frame, stream=True, verbose=False)

#     center_x, center_y = -1, -1
#     frame_center_x, frame_center_y = frame.shape[1] // 2, frame.shape[0] // 2
#     most_closed_center_x = float('inf')

#     for result in results:
#         for box in result.boxes:
#             pos = box.xyxy[0].tolist()
#             x1, y1, x2, y2 = map(round, pos)
#             class_id = int(box.cls[0].item())
            
#             if class_id == 0:  # 필요한 클래스 ID만 처리
#                 object_center = ((x1 + x2) // 2, (y1 + y2) // 2)
#                 dist_from_center = np.abs(object_center[0] - frame_center_x)
                
#                 if object_center[0] < w_start or object_center[0] > w_end:
#                     continue
                
#                 if dist_from_center < most_closed_center_x:
#                     most_closed_center_x = dist_from_center
#                     center_x = object_center[0]
#                     center_y = object_center[1]
                    
#                     w_margin = (x2 - x1) * margin_ratio // 2
#                     h_margin = (y2 - y1) * margin_ratio // 2

#                     # 가장 중앙에 가까운 객체만 저장
#                     w_margin_list = [w_margin]
#                     h_margin_list = [h_margin]
                    
#                     detected = True
                    
#                     if save:
#                         save_frame(frame, results)

#     if center_x == -1 or center_y == -1:
#         center_x, center_y = pre_center_x, pre_center_y
#         save_failure_frame(frame)
        
#     return center_x, center_y, detected


def save_frame(frame, results):
    save_dir = os.path.join(os.getcwd(), 'detection_success')
    idx = 0
    image_name = f'img_{idx}.png'
    while os.path.exists(os.path.join(save_dir, image_name)):
        idx += 1
        image_name = f'img_{idx}.png'
    image_path = os.path.join(save_dir, image_name)
    result_img = results.render()[0]
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(image_path, result_img)

def save_failure_frame(frame):
    path = os.path.join(os.getcwd(), 'detection_fail')
    idx = 0
    image_name = f'img_{idx}.png'
    while os.path.exists(os.path.join(path, image_name)):
        idx += 1
        image_name = f'img_{idx}.png'
    image_path = os.path.join(path, image_name)
    cv2.imwrite(image_path, frame)


def object_detection(frame, model, pre_center_x, pre_center_y, show=False, save=False):
    # YOLOv5 모델을 사용하여 객체 인식
    
    w = frame.shape[1]
    w_start = int(w // 4)
    w_end = int(w * 3 // 4)
    w_mid = w // 2
    # slice_frame = frame[:, w_start:w_end]
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame)
    
    # print(results.xyxy[0].cpu().numpy())
    if show:
        # results.save(save_dir='output/yolo_predictions/img.png')
        
        results.show()

    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    detections = results.xyxy[0].cpu().numpy()
    
    center_x, center_y = -1, -1
    
    frame_center_x, frame_center_y = frame.shape[1] // 2, frame.shape[0] // 2
    center_x_list = []
    center_y_list = []
    distance_list = []

    most_center_x = 10000   
    
    for idx, detection in enumerate(detections):
        x1, y1, x2, y2, confidence, class_id = detection
        # 임의로 0번 클래스를 나로 설정
        if int(class_id) == 0: 
            
            object_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            dist_from_pre_center = np.sqrt((pre_center_x - object_center[0])**2 + (pre_center_y - object_center[1])**2)
            
            # 객체의 x 위치가 프레임의 중앙에 있는지 확인: 중앙 = (w//2, h//2)
            # if x1 < w_start or x2 > w_end:
            #    continue
            
            dist_from_center = np.abs(object_center[0] - frame_center_x)
            if dist_from_pre_center > w//3:
                continue
            
            if dist_from_center < most_center_x:
                most_center_x = dist_from_center
                center_x = object_center[0]
                center_y = object_center[1]
                
                w_margin = (x2 - x1)* margin_ratio //2
                h_margin = (y2 - y1)* margin_ratio //2

                w_margin_list.append(w_margin)
                h_margin_list.append(h_margin)
            # 객체의 이전 프레임과의 거리가 500 이상인 경우 이전 프레임의 중심점을 사용
            # if pre_center_x != -1 and pre_center_y != -1:
            #     distance = np.sqrt((pre_center_x - (x1 + x2) // 2)**2 + (pre_center_y - (y1 + y2) // 2)**2)
            #     if distance > 300 :
            #         print(distance)
            #         continue
    
            # # 중심점 계산
            # center_x = object_center[0]
            # center_y = object_center[1]
            
            # w_margin = (x2 - x1)* margin_ratio //2
            # h_margin = (y2 - y1)* margin_ratio //2

            # w_margin_list.append(w_margin)
            # h_margin_list.append(h_margin)
            # break
            
            # for saving frame     
            if save:
                save = False
                save_dir = os.path.join(os.getcwd(), 'detection_success')
                idx = 0
                image_name = f'img_{idx}.png'
                while os.path.exists(os.path.join(save_dir, image_name)):
                    idx += 1
                    image_name = f'img_{idx}.png'
                image_path = os.path.join(save_dir, image_name)
                result_img = results.render()[0]
                result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(image_path, result_img)

    # 프레임에서 객체를 찾지 못한 경우 이전 프레임의 중심점을 사용  
    if center_x == -1 or center_y == -1:
        center_x, center_y = pre_center_x, pre_center_y
        # for data acquisition
        path = os.path.join(os.getcwd(), 'detection_fail')
        idx = 0
        image_name = f'img_{idx}.png'
        while os.path.exists(os.path.join(path, image_name)):
            idx += 1
            image_name = f'img_{idx}.png'
        image_path = os.path.join(path, image_name)
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_path, frame)

    return center_x, center_y


def exponential_moving_average(prices, weighting_factor=0.4):
    ema = np.zeros(len(prices))

    window_size = len(prices) // 100
    
    ema[:window_size] = centered_moving_average(prices[:window_size])

    for i in range(window_size, len(prices)):
        ema[i] = (prices[i] * weighting_factor) + (ema[i - 1] * (1 - weighting_factor))
        # print(i, ema[i])
    return ema


def centered_moving_average(list):
    avg_list = []
    
    # window_size = len(list) // 100
    window_size = 10
    
    for i in range(len(list)):
        start = max(0, i - window_size)
        end = min(len(list), i + window_size)
        avg = np.mean(list[start:end])
        avg_list.append(avg)
        
    return avg_list


def moving_average(list):
    avg_list = []
    window_size = len(list) // 100
    
    for i in range(len(list)):
        start = max(0, i - window_size)
        avg = np.mean(list[start:i+1])
        avg_list.append(avg)
    return avg_list


def draw_pose(frame, landmarks, connections):
    h, w, _ = frame.shape
    for landmark in landmarks.landmark:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    for connection in connections:
        x1 = int(landmarks.landmark[connection[0]].x * w)
        y1 = int(landmarks.landmark[connection[0]].y * h)
        x2 = int(landmarks.landmark[connection[1]].x * w)
        y2 = int(landmarks.landmark[connection[1]].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame

def crop_region_estimation():
    pass

def crop_frame(frame, center_x, center_y, w_margin, h_margin):        
    
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    
    # 자를 영역 계산
    crop_x1 = int(max(center_x - w_margin, 0))
    crop_y1 = int(max(center_y - h_margin, 0))
    crop_x2 = int(min(center_x + w_margin, frame_width))
    crop_y2 = int(min(center_y + h_margin, frame_height))
    
    # 자를 영역이 마진 영역보다 작은 경우, 자를 영역을 마진만큼 옮김
    if crop_x2 - crop_x1 < 2 * w_margin:
        if crop_x1 == 0:
            crop_x2 = crop_x1 + 2 * w_margin
        else:
            crop_x1 = crop_x2 - 2 * w_margin
    if crop_y2 - crop_y1 < 2 * h_margin:
        if crop_y1 == 0:
            crop_y2 = crop_y1 + 2 * h_margin
        else:
            crop_y1 = crop_y2 - 2 * h_margin
    
    # 자른 프레임의 크기가 고정된 크기와 맞지 않으면 패딩 추가
    blank_frame = np.zeros((h_margin * 2, w_margin * 2, 3), dtype=np.uint8)
    cropped_part = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    cropped_h, cropped_w = cropped_part.shape[:2]

    # 자른 프레임을 중앙에 배치
    start_y = (blank_frame.shape[0] - cropped_h) // 2
    start_x = (blank_frame.shape[1] - cropped_w) // 2
    blank_frame[start_y:start_y + cropped_h, start_x:start_x + cropped_w] = cropped_part
    cropped_frame = cv2.cvtColor(blank_frame, cv2.COLOR_RGB2BGR)
    
    return cropped_frame
    
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def main(file_name=None, dir_name=None, model=None, version=5):
    """ Configuration """
    # Pose estimation을 하는 경우
    do_pose_estimation = False
    # 객체 인식 결과를 보여줄지 여부
    show_object_detection = False
    # 결과를 저장할지 여부
    save = False
    # path specification
    current_path = os.getcwd()
    
    output_dir = dir_name.split('/')[1]
    os.makedirs(f'output/{output_dir}', exist_ok=True)
    input_video_path = os.path.join(current_path, dir_name, file_name)
    output_video_path = os.path.join(current_path, f'output/{output_dir}', f'test.mp4')

    if not os.path.isfile(input_video_path):
        print("Input 경로가 존재하지 않습니다.")
        print(f"지정한 경로: {input_video_path}")
        return 
    # 잘라낼 프레임의 x, y 구간 설정 (예: 중심에서 100px 이내)
    # w_margin = int(1080 * 0.8) // 2 
    # h_margin = int(1080 * 0.8) // 2 

    """ Configuration """
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(input_video_path)

    # 비디오 프레임 정보 가져오기
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print("비디오 정보")
    print("비디오이름: ", file_name)
    print("프레임 크기: ", frame_width, frame_height)
    print(f"전체 프레임 길이: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}, 동영상 길이: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)}초")

    center_x_list = []
    center_y_list = []
    
    print("비디오 편집 시작")

    # 비디오 편집 시작
    pre_center_x, pre_center_y = -1, -1
    cnt = 0
    sampling_rate = sampling_rate_default
    start = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        # cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)  
        if not ret:
            break
        
        if cnt % sampling_rate == 0:
            if version == 5:
                center_x, center_y = object_detection(frame, model, pre_center_x, pre_center_y, show=show_object_detection, save=save)
            elif version == 8:
                center_x, center_y = yolov8(frame, model, pre_center_x, pre_center_y)
                # if not detected:
                #     sampling_rate = sampling_rate_tight
                # else:
                #     sampling_rate = sampling_rate_default
            # continue
        
        if center_x == -1 or center_y == -1:
            center_x, center_y = int(frame_width // 2), int(frame_height // 2)
        
        center_x_list.append(center_x)
        center_y_list.append(center_y)
        pre_center_x, pre_center_y = center_x, center_y
        cnt += 1

        if cnt % 100 == 0:
            print("processing frame: ", cnt)
    cap.release()
    del(model)
    print(f'{(time.time() - start):.01f}초 소요')
    
    new_x_list = centered_moving_average(center_x_list)
    new_y_list = centered_moving_average(center_y_list)
    
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(center_x_list)
    ax[0].plot(new_x_list)
    ax[0].set_title('Center X Coordinates')
    ax[1].plot(center_y_list)
    ax[1].plot(new_y_list)
    ax[1].set_title('Center Y Coordinates')
    plt.savefig('CMA_center_sw.png')

    # print(len(center_x_list), len(new_x_list))
    
    # 프레임 저장 리스트
    frames = []
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(input_video_path)
    
    w_margin = int(np.mean(w_margin_list))
    h_margin = int(np.mean(h_margin_list))
    
    # 마진이 너무 작은 경우, 최소 마진을 설정
    if w_margin < frame_width//3 or h_margin < frame_height//3:
        w_margin = frame_width//3
        h_margin = frame_height//3
    
    # 정방형 모드일 경우
    if square_mode:
        margin = max(w_margin, h_margin)
        w_margin = margin
        h_margin = margin
    
    for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if not ret:
            break
        
        # 프레임 자르기
        cropped_frame = crop_frame(frame, new_x_list[i], new_y_list[i], w_margin, h_margin)
        
        # 자른 프레임을 프레임 리스트에 추가
        frames.append(cropped_frame)
        
        if i % 100 == 0:
            print("processing frame: ", i)
    cap.release()
    
    # Pose estimation 수행시
    if do_pose_estimation:
        # MediaPipe Pose 모델 초기화
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)
        
        for i in range(len(frames)):
            frame = frames[i]
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                frame = draw_pose(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            frame[i] = frame

    # 비디오 클립 생성 및 저장
    clip = ImageSequenceClip(frames, fps=fps,)
    
    # 비디오에 오디오 추가
    audio = VideoFileClip(input_video_path).audio
    # audio.write_audiofile('temp_audio.mp3', verbose=False)
    # audio = AudioFileClip('temp_audio.mp3')
    clip = clip.set_audio(audio)
    
    # 비디오 저장
    clip.write_videofile(output_video_path, codec='libx264', audio_codec='aac', verbose = False)
    # os.remove('temp_audio.mp3')
    print("비디오 편집 완료: ", output_video_path)
    print("출력 영상 크기:", w_margin * 2, h_margin * 2)

def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, default='/Users/tamtam211/ocae/cruxcut_v001/input/testbench/sh_theclimb_ilsan_p.mp4', help='input path')
    parser.add_argument('--folder', '-f', type=str, default='0808', help='folder name')
    parser.add_argument('--version', '-v', type=int, default=8, help='YOLO version')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = getargs()
    
    # 사용할 모델 선택
    version = args.version
    
    # input path
    input_path = args.input
    folder = args.folder

    # 현재 시스템에서 사용 가능한 디바이스 확인
    device = get_device()
    # device = "cpu"
    print(f"Using device: {device}")
    
    file_path = "input/0725/20240725_203529.mp4"
    
    # YOLOv8 모델 로드
    if version == 8:
        model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
        model.fuse()  # fuse model (recommended for improved inference speed)
        model.to(device)  # load to device (recommended for improved inference speed)
    # main(file_path, model)

    # YOLOv5 모델 로드
    elif version == 5:
        # model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, skip_validation=True)
        model = YOLO("yolov5s.pt")
        model.fuse()
        model.conf = 0.3
        model.classes = [0]

    # model.predict(file_path, stream=False, show=True, save=False, save_dir='output')
    # model.track(source=file_path, show=True, classes= [0], tracker ="bytetrack.yaml")
    # results = model.track(source=file_path, show=True, classes= [0], tracker ="bytetrack.yaml")
    
    dir_name = os.path.join(input_path, folder)
    
    # # # for a folder
    # for filename in os.listdir(dir_name):
    #     if filename.endswith('.mp4') or filename.endswith('.mov'):
    #         main(filename, dir_name, model, version=version)

    # for a single video
    video_name = 'enbi.mp4'
    main(video_name, dir_name, model=model, version=version)
    