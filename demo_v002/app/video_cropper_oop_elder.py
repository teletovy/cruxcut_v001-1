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
from dataclasses import dataclass
from typing import List, Tuple, Optional
import logging
# from profiler import Profiler

@dataclass
class CropperConfig:
    sampling_rate_default: int = 30
    sampling_rate_tight: int = 5
    margin_ratio: float = 1.6
    square_mode: bool = True
    do_pose_estimation: bool = False
    show_object_detection: bool = False
    save_detection: bool = False
    model_version: int = 8
    confidence_threshold: float = 0.3

class VideoCropper:
    def __init__(self, config: Optional[CropperConfig] = None):
        self.config = config or CropperConfig()
        self.model = None
        self.device = self._get_device()
        self.detector = None
        self.processor = None
        
    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
        
    def load_model(self):
        if self.config.model_version == 8:
            self.model = YOLO("yolov8n.pt")
        elif self.config.model_version == 5:
            self.model = YOLO("yolov5s.pt")
            self.model.conf = self.config.confidence_threshold
            self.model.classes = [0]
        
        self.model.fuse()
        self.model.to(self.device)
        self.detector = ObjectDetector(self.model, self.config)
        
    def process_video(self, input_path: str, output_path: str) -> bool:
        self.processor = VideoProcessor(input_path, output_path, self.config)
        self.processor.process_video(self.detector)
        return True

class ObjectDetector:
    def __init__(self, model: YOLO, config: CropperConfig):
        self.model = model
        self.config = config
        self.w_margin_list: List[float] = []
        self.h_margin_list: List[float] = []
        
    def detect_person(self, frame: np.ndarray, pre_center_x: float, pre_center_y: float) -> Tuple[float, float]:
        w = frame.shape[1]
        w_start = int(w // 4)
        w_end = int(w * 3 // 4)
        frame_center_x = frame.shape[1] // 2
        
        results = self.model(frame, stream=True, verbose=False)
        center_x, center_y = -1, -1
        most_closed_center_x = float('inf')
        
        for result in results:
            for box in result.boxes:
                pos = box.xyxy[0].tolist()
                pos = [round(x) for x in pos]
                x1, y1, x2, y2 = pos
                class_id = int(box.cls[0].item())
                
                if class_id == 0:  # Person class
                    object_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    dist_from_pre_center = np.sqrt(
                        (pre_center_x - object_center[0])**2 + 
                        (pre_center_y - object_center[1])**2
                    )
                    
                    if object_center[0] < w_start or object_center[0] > w_end:
                        continue
                        
                    dist_from_center = np.abs(object_center[0] - frame_center_x)
                    if dist_from_pre_center > w//3:
                        continue
                    
                    if dist_from_center < most_closed_center_x:
                        most_closed_center_x = dist_from_center
                        center_x = object_center[0]
                        center_y = object_center[1]
                        
                        w_margin = (x2 - x1) * self.config.margin_ratio // 2
                        h_margin = (y2 - y1) * self.config.margin_ratio // 2
                        
                        self.w_margin_list.append(w_margin)
                        self.h_margin_list.append(h_margin)
                        
        if center_x == -1 or center_y == -1:
            center_x, center_y = pre_center_x, pre_center_y
            
        return center_x, center_y

class TrajectorySmoothing:
    @staticmethod
    def centered_moving_average(data: List[float], window_size: int = 10) -> List[float]:
        avg_list = []
        for i in range(len(data)):
            start = max(0, i - window_size)
            end = min(len(data), i + window_size)
            avg = np.mean(data[start:end])
            avg_list.append(avg)
        return avg_list

class VideoProcessor:
    def __init__(self, input_path: str, output_path: str, config: CropperConfig):
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
        # self.profiler = Profiler()
        
        
    def _init_video_capture(self):
        self.cap = cv2.VideoCapture(self.input_path)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
    def _crop_frame(self, frame: np.ndarray, center_x: float, center_y: float, 
                   w_margin: float, h_margin: float) -> np.ndarray:
        frame_height, frame_width = frame.shape[:2]
        
        crop_x1 = int(max(center_x - w_margin, 0))
        crop_y1 = int(max(center_y - h_margin, 0))
        crop_x2 = int(min(center_x + w_margin, frame_width))
        crop_y2 = int(min(center_y + h_margin, frame_height))
        
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
        
        blank_frame = np.zeros((int(h_margin * 2), int(w_margin * 2), 3), dtype=np.uint8)
        cropped_part = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        cropped_h, cropped_w = cropped_part.shape[:2]
        
        start_y = (blank_frame.shape[0] - cropped_h) // 2
        start_x = (blank_frame.shape[1] - cropped_w) // 2
        blank_frame[start_y:start_y + cropped_h, start_x:start_x + cropped_w] = cropped_part
        
        return cv2.cvtColor(blank_frame, cv2.COLOR_RGB2BGR)
    
    def process_video(self, detector: ObjectDetector):
        self._init_video_capture()
        logging.info(f"Processing video: {self.input_path}")
        logging.info(f"Frame size: {self.frame_width}x{self.frame_height}, FPS: {self.fps}")
        
        pre_center_x, pre_center_y = -1, -1
        frame_count = 0
        start_time = time.time()
        
        # self.profiler.start()
        # First pass: Detect person and collect center points
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            if frame_count % self.config.sampling_rate_default == 0:
                center_x, center_y = detector.detect_person(frame, pre_center_x, pre_center_y)
                # self.profiler.log_memory_usage(f"After detection frame {frame_count}")
                
            if center_x == -1 or center_y == -1:
                center_x, center_y = self.frame_width // 2, self.frame_height // 2
                
            self.center_x_list.append(center_x)
            self.center_y_list.append(center_y)
            pre_center_x, pre_center_y = center_x, center_y
            
            if frame_count % 100 == 0:
                logging.info(f"Processing frame: {frame_count}")
                
            frame_count += 1
            
        self.cap.release()
        
        # first_pass_metrics = self.profiler.stop()
        logging.info("\nFirst pass metrics:")
        # logging.info(f"Time taken: {first_pass_metrics['execution_time']:.2f} seconds")
        # logging.info(f"Memory used: {first_pass_metrics['memory_used']:.2f} MB")
        # logging.info(f"Peak memory: {first_pass_metrics['peak_memory']:.2f} MB")
        
        # self.profiler.start()
        
        # Smooth trajectories
        smoothed_x = TrajectorySmoothing.centered_moving_average(self.center_x_list)
        smoothed_y = TrajectorySmoothing.centered_moving_average(self.center_y_list)
        
        # Calculate margins
        w_margin = int(np.mean(detector.w_margin_list))
        h_margin = int(np.mean(detector.h_margin_list))
        
        # Ensure minimum margins
        if w_margin < self.frame_width//3 or h_margin < self.frame_height//3:
            w_margin = self.frame_width//3
            h_margin = self.frame_height//3
            
        if self.config.square_mode:
            margin = max(w_margin, h_margin)
            w_margin = h_margin = margin
            
        # Second pass: Crop and save frames
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
            
            if frame_count % 100 == 0:
                logging.info(f"Cropping frame: {frame_count}")
                # self.profiler.log_memory_usage(f"After cropping frame {frame_count}")
                
            frame_count += 1
            
        self.cap.release()
        
        # second_pass_metrics = self.profiler.stop()
        logging.info("\nSecond pass metrics:")
        # logging.info(f"Time taken: {second_pass_metrics['execution_time']:.2f} seconds")
        # logging.info(f"Memory used: {second_pass_metrics['memory_used']:.2f} MB")
        # logging.info(f"Peak memory: {second_pass_metrics['peak_memory']:.2f} MB")
        
        # Save video
        # self.profiler.start()
        self._save_video()

        return True
        # save_metrics = self.profiler.stop()
        
        logging.info("\nVideo saving metrics:")
        # logging.info(f"Time taken: {save_metrics['execution_time']:.2f} seconds")
        # logging.info(f"Memory used: {save_metrics['memory_used']:.2f} MB")
        # logging.info(f"Peak memory: {save_metrics['peak_memory']:.2f} MB")
        
    def _save_video(self):
        clip = ImageSequenceClip(self.frames, fps=self.fps)
        
        # Add audio
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
        logging.info(f"Video saved to: {self.output_path}")
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True, help='Input video path')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output video path')
    parser.add_argument('--version', '-v', type=int, default=8, help='YOLO version (5 or 8)')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize configuration
    config = CropperConfig(model_version=args.version)
    
    # Create and setup video cropper
    cropper = VideoCropper(config)
    cropper.load_model()
    
    # Process video
    cropper.process_video(args.input, args.output)

# Initialize configuration
config = CropperConfig(model_version=8)

# Create and setup video cropper
cropper = VideoCropper(config)
cropper.load_model()


if __name__ == '__main__':
    main() 