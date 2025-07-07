import cv2
from ultralytics import YOLO
import os
import time

class VideoDetector:
    def __init__(self, model_name='yolo11n.pt'):
        # YOLOv8 모델 로드
        self.model = YOLO(model_name)

    def process_video(self, input_path, output_path):
        try:
            # 영상 파일 열기
            cap = cv2.VideoCapture(input_path)
            
            # 비디오 속성 가져오기
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 출력 비디오 설정
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 객체 감지
                results = self.model(frame)
                
                # 감지 결과 그리기
                for result in results:
                    annotated_frame = result.plot()
                
                # 결과 프레임을 파일에 저장
                out.write(annotated_frame)
                
                frame_count += 1
                
            cap.release()
            out.release()
            
            return True, frame_count, total_frames
            
        except Exception as e:
            print(f"Error processing video: {str(e)}")
            return False, 0, 0

# 전역 detector 인스턴스 생성
detector = VideoDetector('yolov8n.pt')


if __name__ == "__main__":
    input_path = "input.mp4"
    output_path = "output.mp4"
    model_name = 'yolo11n.pt' # yolo model names
    
    detector.process_video(input_path, output_path)
