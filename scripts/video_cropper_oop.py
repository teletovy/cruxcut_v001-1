import cv2
import torch
import os
import matplotlib.pyplot as plt
from moviepy.editor import *
import numpy as np
import mediapipe as mp
import time
from ultralytics import YOLO, YOLOWorld
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import logging
# from profiler import Profiler
import multiprocessing

@dataclass
class CropperConfig:
    # 비디오 샘플링 기본 프레임 간격 (기본 탐지 주기)
    sampling_rate_default: int = 30
    # 타이트 샘플링(더 촘촘한 탐지 주기, 필요시 사용)
    sampling_rate_tight: int = 5
    # 마진 비율(탐지된 바운딩 박스에 곱해 크롭 영역 확장)
    margin_ratio: float = 1.6
    # 정사각형 크롭 모드 여부(True면 정사각형으로 크롭)
    square_mode: bool = True
    # 포즈 추정 사용 여부(현재 미사용)
    do_pose_estimation: bool = False
    # 객체 탐지 결과 시각화 여부
    show_object_detection: bool = False
    # 탐지 이미지 저장 여부
    save_detection: bool = True
    # 사용할 YOLO 모델 버전(5, 8, 11 등)
    model_version: int = 11
    # 객체 탐지 신뢰도 임계값
    confidence_threshold: float = 0.3
    
class VideoCropper:
    """
    VideoCropper 클래스
    - 비디오 크롭 파이프라인 전체를 관리하는 클래스입니다.
    - 설정(config)과 모델, 탐지기, 프로세서를 관리합니다.
    """
    def __init__(self, config: Optional[CropperConfig] =  None):
        # 설정값 저장
        self.config = config or CropperConfig()
        self.model = None
        self.device = self._get_device()
        self.detector = None
        self.processor = None
        
    def _get_device(self) -> torch.device:
        # 사용 가능한 디바이스(CUDA/MPS/CPU) 자동 선택
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
        
    def load_model(self):
        """
        YOLO 모델을 설정값에 따라 불러오고, ObjectDetector를 초기화합니다.
        """
        if self.config.model_version == 8:
            if self.model is not None:
                logging.warning("YOLO model already loaded. Skipping reload.")
                return
            self.model = YOLO("yolov8s.pt")
        elif self.config.model_version == 5:
            if self.model is not None:
                logging.warning("YOLO model already loaded. Skipping reload.")
                return
            self.model = YOLO("yolov5s.pt")
            self.model.conf = self.config.confidence_threshold
            self.model.classes = [0]
        elif self.config.model_version == 11:
            if self.model is not None:
                logging.warning("YOLO model already loaded. Skipping reload.")
                return
            self.model = YOLO("yolo11s.pt")
        elif self.config.model_version == -1:
            if self.model is not None:
                logging.warning("YOLO model already loaded. Skipping reload.")
                return
            self.model = YOLOWorld("yolov8m-worldv2.pt")
            self.model.set_classes(["person", "climber"])
        
        self.model.fuse()
        self.model.to(self.device)
        logging.warning(f"Loaded YOLO model version {self.config.model_version} on {self.device}")
        self.detector = ObjectDetector(self.model, self.config)
        
    def process_video(self, input_path: str, output_path: str, job_status: Optional[dict]=None, jobid: Optional[str]=None) -> bool:
        """
        비디오 크롭 전체 파이프라인 실행 함수
        - 입력 비디오를 받아 탐지, 트래킹, 스무딩, 크롭, 저장까지 수행합니다.
        """
        self.processor = VideoProcessor(input_path, output_path, job_status, jobid, self.config)
        self.processor.process_video(self.detector)
        return True

class ObjectDetector:
    """
    ObjectDetector 클래스
    - YOLO 모델을 이용해 프레임에서 사람(클래스 0) 객체를 탐지합니다.
    - 탐지된 중심점 좌표와 마진 정보를 기록합니다.
    """
    def __init__(self, model: YOLO, config: CropperConfig):
        # YOLO 모델과 설정값 저장
        self.model = model
        self.config = config
        # 각 프레임별 마진(너비, 높이) 리스트
        self.w_margin_list: List[float] = []
        self.h_margin_list: List[float] = []
        
    def detect_person(self, frame: np.ndarray, prev_center_x: float, prev_center_y: float, frame_index: int, job_id: str) -> Tuple[float, float]:
        """
        한 프레임에서 사람 객체를 탐지하여 중심 좌표와 마진을 반환합니다.
        - 중심에 가까운 사람 1명을 선택합니다.
        - 탐지 실패 시 이전 중심 좌표를 반환합니다.
        - 탐지 결과 이미지를 저장할 수 있습니다.
        """
        # 프레임 너비 기반 탐지 가능한 중심 영역 정의
        frame_width = frame.shape[1]
        focus_zone_start = frame_width // 6
        focus_zone_end = frame_width * 5 // 6
        frame_center_x = frame_width // 2

        # YOLO 객체 탐지 실행
        results = self.model(frame, stream=True, verbose=False)
        selected_center_x, selected_center_y = -1, -1
        min_distance_to_center = float('inf')

        person_detected = False
        img = frame  # 기본값: 원본 프레임
        # 탐지 결과 반복
        for result in results:
            for bbox in result.boxes:
                # 바운딩 박스 좌표 및 클래스 정보 추출
                x1, y1, x2, y2 = map(round, bbox.xyxy[0].tolist())
                class_id = int(bbox.cls[0].item())
                confidence = bbox.conf[0].item()

                if class_id == 0 or class_id == 1:  # 사람이 탐지된 경우
                    logging.warning(f"Detected person at {(x1, y1, x2, y2)} with confidence {confidence}")

                    # 사람 중심 좌표 계산
                    person_center_x = (x1 + x2) // 2
                    person_center_y = (y1 + y2) // 2

                    # 이전 중심 좌표와의 거리 계산
                    dist_to_prev = np.hypot(person_center_x - prev_center_x, person_center_y - prev_center_y)
                    dist_to_frame_center = abs(person_center_x - frame_center_x)

                    # 탐지 범위 바깥에 있거나 너무 튄 경우 무시
                    if not (focus_zone_start <= person_center_x <= focus_zone_end):
                        continue
                    # if dist_to_prev > frame_width // 4:
                    #     continue

                    # 중심에 가장 가까운 사람 하나만 선택
                    if dist_to_frame_center < min_distance_to_center:
                        min_distance_to_center = dist_to_frame_center
                        selected_center_x = person_center_x
                        selected_center_y = person_center_y
                        selected_bbox = (x1, y1, x2, y2)

                        # 마진 계산 및 저장
                        width_margin = ((x2 - x1) * self.config.margin_ratio) // 2
                        height_margin = ((y2 - y1) * self.config.margin_ratio) // 2

                        person_detected = True
                        
        if person_detected:
            # 마진 리스트에 추가
            self.w_margin_list.append(width_margin)
            self.h_margin_list.append(height_margin)

        # 옵션에 따라 탐지 이미지 저장
        if self.config.save_detection:
            annotated = result.plot()
            img = annotated.copy()

            cv2.line(annotated, (frame_center_x, 0), (frame_center_x, frame.shape[0]), (255, 0, 0), 2)
            # 필터링 영역 표시
            cv2.line(annotated, (focus_zone_start, 0), (focus_zone_start, frame.shape[0]), (0, 255, 0), 2)
            cv2.line(annotated, (focus_zone_end, 0), (focus_zone_end, frame.shape[0]), (0, 255, 0), 2)  
            # 이전 중심 좌표 표시
            if prev_center_x != -1 and prev_center_y != -1:
                cv2.circle(annotated, (int(prev_center_x), int(prev_center_y)), 5, (0, 255, 0), -1)
                cv2.putText(annotated, f"Prev center: ({int(prev_center_x)}, {int(prev_center_y)})", 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if person_detected:
                # 탐지된 사람 중심에 원 표시
                cv2.circle(annotated, (selected_center_x, selected_center_y), 5, (0, 0, 255), -1)
                # 바운딩 박스 표시
                cv2.rectangle(annotated, (selected_bbox[0], selected_bbox[1]), 
                              (selected_bbox[2], selected_bbox[3]), (0, 0, 255), 5)
                cv2.putText(annotated, f"Person detected at ({selected_center_x}, {selected_center_y})", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                output_path = f"temp/{job_id}/{frame_index}_person_detected.jpg"
                cv2.imwrite(output_path, annotated)
                logging.warning(f"Detection saved to {output_path}")
            else:
                # 사람이 탐지되지 않은 경우에도 탐지 결과 저장
                cv2.putText(annotated, "No person detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # 탐지 결과 이미지 저장
                output_path = f"temp/{job_id}/{frame_index}_no_person_detected.jpg"
                cv2.imwrite(output_path, annotated)
                logging.warning(f"No person detection saved to {output_path}")
            realtime_output = f"detection_output.jpg"
            cv2.imwrite(realtime_output, annotated)
            logging.warning(f"Realtime detection saved to {realtime_output}")

        # 탐지 실패 시 이전 중심 유지
        if selected_center_x == -1 or selected_center_y == -1:
            selected_center_x, selected_center_y = prev_center_x, prev_center_y

        return selected_center_x, selected_center_y, img
    
class TrajectorySmoothing:
    """
    TrajectorySmoothing 클래스
    - 중심점 좌표 리스트에 centered moving average(중앙 이동평균)로 스무딩 처리를 수행합니다.
    """
    @staticmethod
    def centered_moving_average(data: List[float], window_size: int = 5) -> List[float]:
        """
        중심 이동평균을 적용하여 좌표의 급격한 변화(진동 등)를 부드럽게 만듭니다.
        """
        avg_list = []
        for i in range(len(data)):
            start = max(0, i - window_size)
            end = min(len(data), i + window_size)
            avg = np.mean(data[start:end])
            avg_list.append(avg)
        return avg_list
    
    def KalmanFilter(data: List[float], process_variance: float = 1e-5, measurement_variance: float = 1e-1) -> List[float]:
        """
        칼만 필터를 적용하여 좌표의 노이즈를 제거합니다.
        """
        n = len(data)
        x = np.zeros(n)
        P = np.zeros(n)
        Q = process_variance
        R = measurement_variance
        
        for i in range(n):
            # 예측 단계
            if i == 0:
                x[i] = data[i]
                P[i] = 1.0
            else:
                x[i] = x[i-1]
                P[i] = P[i-1] + Q
            
            # 업데이트 단계
            K = P[i] / (P[i] + R)
            x[i] += K * (data[i] - x[i])
            P[i] *= (1 - K)
        
        return x.tolist()

class VideoProcessor:
    """
    VideoProcessor 클래스
    - 비디오의 프레임을 읽고, 탐지/스무딩/크롭/저장 전체 파이프라인을 담당합니다.
    - 프레임별 중심점 및 마진 계산, 트래킹, 최종 비디오 저장까지 수행합니다.
    """
    def __init__(self, input_path: str, output_path: str, job_status: dict, jobid: str, config: CropperConfig):
        # 비디오 입출력 경로, 설정, 상태 관리 등 저장
        self.input_path = input_path
        self.output_path = output_path
        self.config = config
        self.frames = []
        self.center_x_list = []
        self.center_y_list = []
        self.cap = None
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 0
        self.job_status = job_status
        self.job_id = jobid
        # self.profiler = Profiler()
        
    def _init_video_capture(self):
        """
        비디오 캡처 객체를 초기화하고, 프레임 해상도/프레임수/FPS 등 정보를 읽어옵니다.
        """
        self.cap = cv2.VideoCapture(self.input_path)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.job_status is not None:
            # 작업 상태  c     
            self.job_status[self.job_id] = {
                "status": "processing",
                "progress": 0,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "input_file": self.input_path,
                "output_file": self.output_path,
            }

    def _crop_frame(self, frame: np.ndarray, center_x: float, center_y: float, 
                   w_margin: float, h_margin: float) -> np.ndarray:
        """
        입력 프레임에서 중심점과 마진을 기준으로 크롭을 수행하고, 크롭 영역이 부족하면 블랭크 패딩을 추가합니다.
        """
        frame_height, frame_width = frame.shape[:2]
        
        crop_x1 = int(max(center_x - w_margin, 0))
        crop_y1 = int(max(center_y - h_margin, 0))
        crop_x2 = int(min(center_x + w_margin, frame_width))
        crop_y2 = int(min(center_y + h_margin, frame_height))
        
        # 크롭 영역이 마진보다 작으면 보정
        if crop_x2 - crop_x1 < 2 * w_margin:
            if crop_x1 == 0:
                crop_x2 = crop_x1 + int(2 * w_margin)
            else:
                crop_x1 = crop_x2 - int(2 * w_margin)
                
        if crop_y2 - crop_y1 < 2 * h_margin:
            if crop_y1 == 0:
                crop_y2 = crop_y1 + int(2 * h_margin)
            else:
                crop_y1 = crop_y2 - int(2 * h_margin)
        
        # 블랭크(검정색) 프레임 생성 후, 크롭 영역을 정중앙에 삽입
        blank_frame = np.zeros((int(h_margin * 2), int(w_margin * 2), 3), dtype=np.uint8)
        cropped_part = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        cropped_h, cropped_w = cropped_part.shape[:2]
        
        start_y = (blank_frame.shape[0] - cropped_h) // 2
        start_x = (blank_frame.shape[1] - cropped_w) // 2
        blank_frame[start_y:start_y + cropped_h, start_x:start_x + cropped_w] = cropped_part
        
        return cv2.cvtColor(blank_frame, cv2.COLOR_RGB2BGR)
    
    def process_video(self, detector: ObjectDetector):
        """
        비디오 전체 처리 파이프라인
        1. 프레임별 사람 탐지 및 중심점 좌표 수집
        2. 중심점 트래젝토리 스무딩
        3. 마진 계산
        4. 프레임 크롭
        5. 최종 비디오 저장
        """
        self._init_video_capture()
        logging.warning(f"Processing video: {self.input_path}")
        logging.warning(f"Frame size: {self.frame_width}x{self.frame_height}, FPS: {self.fps}")
        
        os.makedirs(f"temp/{self.job_id}", exist_ok=True)
        
        pre_center_x, pre_center_y = -1, -1
        frame_count = 0
        start_time = time.time()
        
        # 1. 첫 번째 패스: 탐지 및 중심점 좌표 수집
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # 초반 프레임은 탐지 간격을 짧게 설정, 이후에는 기본 탐지 간격(sampling_rate_default) 사용
            if  frame_count < 90 or frame_count % self.config.sampling_rate_default == 0:
                center_x, center_y, detected_img = detector.detect_person(frame, pre_center_x, pre_center_y, frame_count, self.job_id)
                logging.warning(f"center_x:{center_x}, center_y:{center_y}")
            # 탐지 실패 시 프레임 중앙 사용
            if center_x == -1 or center_y == -1:
                center_x, center_y = self.frame_width // 2, self.frame_height // 2
            self.center_x_list.append(center_x)
            self.center_y_list.append(center_y)
            pre_center_x, pre_center_y = center_x, center_y
            
            # 프레임 저장
            if frame_count % 30 == 0:
                cv2.imwrite(f"temp/{self.job_id}/frame_{frame_count}.jpg", frame)  
            
            if frame_count % 100 == 0:
                logging.warning(f"Processing frame: {frame_count}")
                
            # 진행률 갱신(1차 패스: 0~50%)
            progress = int((frame_count / self.total_frames) * 50) 
            if self.job_status is not None:
                self.job_status[self.job_id]["status"] = "processing"
                self.job_status[self.job_id]["progress"] = min(progress, 50)
            # self.profiler.log_memory_usage(f"After processing frame {frame_count}")
            frame_count += 1
        self.cap.release()
        logging.warning("\nFirst pass metrics:")
        
        smoother = TrajectorySmoothing()
        # 2. 중심점 트래젝토리 스무딩
        smoothed_x = TrajectorySmoothing.centered_moving_average(self.center_x_list)
        smoothed_y = TrajectorySmoothing.centered_moving_average(self.center_y_list)
        
        # smoothed_x = TrajectorySmoothing.KalmanFilter(self.center_x_list)
        # smoothed_y = TrajectorySmoothing.KalmanFilter(self.center_y_list)
        
        # 3. 스무딩 좌표에 가우시안 노이즈 추가(더 부드럽게)
        noise_std = 0.0005 * (self.frame_width + self.frame_height) / 2
        smoothed_x = np.array(smoothed_x) + np.random.normal(0, noise_std, len(smoothed_x))
        smoothed_y = np.array(smoothed_y) + np.random.normal(0, noise_std, len(smoothed_y))
        smoothed_x = np.clip(smoothed_x, 0, self.frame_width - 1)
        smoothed_y = np.clip(smoothed_y, 0, self.frame_height - 1)
        logging.warning(f"Trajectory smoothing completed in {time.time() - start_time:.2f} seconds")
        logging.warning(f"Total frames processed: {frame_count}")
        
        print(len(detector.w_margin_list), len(detector.h_margin_list))
        
        # 4. 마진 계산(탐지 결과 기반, 없으면 기본값 사용)
        if not detector.w_margin_list or not detector.h_margin_list:
            logging.warning("No valid margins detected. Using default margins.")
            detector.w_margin_list = [self.frame_width // 4]
            detector.h_margin_list = [self.frame_height // 4]
        if np.isnan(detector.w_margin_list).any() or np.isnan(detector.h_margin_list).any():
            logging.warning("NaN values found in margins. Using default margins.")
            detector.w_margin_list = [self.frame_width // 4]
            detector.h_margin_list = [self.frame_height // 4]
        
        w_margin = int(np.mean(detector.w_margin_list))
        h_margin = int(np.mean(detector.h_margin_list))
        
        # 최소 마진 보장
        if w_margin < self.frame_width//4 or h_margin < self.frame_height//4:
            w_margin = self.frame_width//4
            h_margin = self.frame_height//4
        # 최대 마진 보장
        if w_margin > self.frame_width//3 or h_margin > self.frame_height//3:
            w_margin = self.frame_width//3
            h_margin = self.frame_height//3
            
        # 정사각형 모드면 마진 통일
        if self.config.square_mode:
            margin = max(w_margin, h_margin)
            w_margin = h_margin = margin
        
        logging.warning(f"Calculated margins: width={w_margin}, height={h_margin}")
            
        # if self.config.save_detection:
        #     from matplotlib import pyplot as plt
        #     # 스무딩된 좌표 시각화
        #     fig, ax = plt.subplots(2, 1, figsize=(12, 6))
        #     ax[0].plot(self.center_x_list, label='Original X', alpha=0.5)
        #     ax[0].plot(smoothed_x, label='Smoothed X', color='orange')
        #     ax[0].set_title('X Coordinate Smoothing')
        #     ax[0].legend()
        #     ax[1].plot(self.center_y_list, label='Original Y', alpha=0.5)
        #     ax[1].plot(smoothed_y, label='Smoothed Y', color='orange')
        #     ax[1].set_title('Y Coordinate Smoothing')
        #     ax[1].legend()
        #     plt.tight_layout()
        #     plt.savefig(f"temp/{self.job_id}/Trajectory_plots.png")
        #     plt.clf()

        #     # 마진 리스트 시각화
        #     fig, ax = plt.subplots(2, 1, figsize=(8, 4))
        #     ax[0].plot(detector.w_margin_list, label='Width Margin', alpha=0.5)    
        #     ax[1].plot(detector.h_margin_list, label='Height Margin', alpha=0.5)
        #     ax[0].axhline(w_margin, 0, len(detector.w_margin_list), color='red', linestyle='--', label='Fixed Margin')
        #     ax[1].axhline(h_margin, 0, len(detector.h_margin_list), color='red', linestyle='--', label='Fixed Margin')
        #     ax[0].set_title(f'Width Margin Over Frames (fixed: {w_margin})')
        #     ax[1].set_title(f'Height Margin Over Frames(fixed: {h_margin})')
        #     ax[0].legend()
        #     ax[1].legend()
        #     plt.tight_layout()
        #     plt.savefig(f"temp/{self.job_id}/Margin_plots.png")
        #     plt.close(fig)
        #     plt.clf()
            
        # 5. 두 번째 패스: 스무딩된 좌표로 프레임 크롭 및 저장
        self.cap = cv2.VideoCapture(self.input_path)
        frame_count = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            cropped_frame = self._crop_frame(
                frame, smoothed_x[frame_count], smoothed_y[frame_count], 
                w_margin, h_margin
            )
            self.frames.append(cropped_frame)

            # 크롭된 프레임 저장
            if frame_count % 30 == 0:
                cv2.imwrite(f"temp/{self.job_id}/cropped_frame_{frame_count}.jpg", cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))  
            
            if frame_count % 100 == 0:
                logging.warning(f"Cropping frame: {frame_count}")
                # self.profiler.log_memory_usage(f"After cropping frame {frame_count}")
                
            # 진행률 갱신(2차 패스: 50~100%)
            if self.job_status is not None:
                self.job_status[self.job_id]["status"] = "processing"
                progress = int((frame_count / self.total_frames) * 50) + 50
                self.job_status[self.job_id]["progress"] = min(progress, 100)
            # self.profiler.log_memory_usage(f"After processing frame {frame_count}")
            frame_count += 1
            
        self.cap.release()
        
        logging.warning("\nSecond pass metrics:")
        
        # 6. 최종 비디오 저장
        self._save_video()

        logging.warning("\nVideo saving metrics:")
        
    def _save_video(self):
        """
        크롭된 프레임 시퀀스를 비디오 파일로 저장하고, 오디오도 함께 붙입니다.
        """
        if self.job_status is not None:
            self.job_status[self.job_id]["status"] = "converting"

        clip = ImageSequenceClip(self.frames, fps=self.fps)
        # 오디오 추가
        try:
            audio = VideoFileClip(self.input_path).audio
            clip = clip.set_audio(audio)
        except Exception as e:
            logging.warning(f"Could not add audio: {str(e)}")
        
        clip.write_videofile(
            self.output_path, 
            codec='libx264',
            audio_codec='aac',
            verbose=False
        )
            
        # 작업 완료 상태 갱신
        if self.job_status is not None:
            self.job_status[self.job_id]["status"] = "completed"
            self.job_status[self.job_id]["progress"] = 100
            self.job_status[self.job_id]["output_file"] = self.output_path
        logging.warning(f"Video saved to: {self.output_path}")


# Global 변수로 CropperConfig와 VideoCropper 인스턴스 생성
# Initialize configuration
config = CropperConfig(model_version=11, save_detection=False, square_mode=False, margin_ratio=3.0, sampling_rate_default=15)
# Create and setup video cropper
cropper = VideoCropper(config)
cropper.load_model()

def main():
    """
    main 함수
    - 커맨드라인 인자 파싱 후 파이프라인 전체 실행
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True, help='Input video path')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output video path')
    parser.add_argument('--version', '-v', type=int, default=11, help='YOLO version (5 or 8 or 11)')
    args = parser.parse_args()
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 설정 초기화
    config.model_version = args.version   
    config.save_detection = True  # 탐지 이미지 저장 여부

    # VideoCropper 생성 및 모델 로드
    cropper = VideoCropper(config)
    cropper.load_model()
    
    # 비디오 처리 실행
    # cropper.process_video(args.input, args.output, jobid = args.input.split('/')[-1].split('.')[0])
    
def experimental_main():
        
    global config, cropper
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', '-v', type=int, default=11, help='YOLO version (5 or 8 or 11)')
    args = parser.parse_args()
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 실험 config 초기화
    config.model_version = args.version
    config.margin_ratio = 2.0
    config.save_detection = True
    config.square_mode = False

    # VideoCropper 생성 및 모델 로드
    cropper.config = config
    cropper.load_model()
    
    input_dir = '20_up'
    output_dir = '20_pr'
    temp_dir = '20_tmp'
    
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    input_path_list = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    # 입력 경로 내의 모든 비디오 파일에 대해 처리 실행
    for input_path in input_path_list:
        output_path = os.path.join(output_dir, f"{os.path.basename(input_path).split('.')[0]}_processed.mp4")
        logging.warning(f"Processing video: {input_path} -> {output_path}")
        cropper.process_video(input_path, output_path, jobid=os.path.basename(input_path).split('.')[0])
        logging.warning(f"Video processing completed: {output_path}")


if __name__ == '__main__':
    # main() 
    experimental_main()