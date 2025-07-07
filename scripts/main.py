from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from fastapi.requests import Request
from fastapi.exceptions import RequestValidationError
import os
import uuid
import shutil
from pathlib import Path
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import json
from datetime import datetime
from typing import Dict, Any
import asyncio
from starlette.exceptions import HTTPException as StarletteHTTPException
from video_cropper_oop import cropper, VideoCropper, CropperConfig  # 비디오 크롭퍼와 설정 클래스 임포트


app = FastAPI(title="CruxCut Video Processing API", version="1.0.0")

@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "details": exc.errors(),
            "body": exc.body
        }
    )
    
# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Next.js 개발 서버
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 디렉토리 설정
UPLOAD_DIR = Path("uploads")
PROCESSED_DIR = Path("processed")
TEMP_DIR = Path("temp")

# 디렉토리 생성
for directory in [UPLOAD_DIR, PROCESSED_DIR, TEMP_DIR]:
    directory.mkdir(exist_ok=True)

# 작업 상태 저장소 (실제 환경에서는 Redis나 데이터베이스 사용)
job_status: Dict[str, Dict[str, Any]] = {}

class VideoProcessor:
    """비디오 처리 클래스"""
    
    @staticmethod
    def detect_climber(frame):
        """클라이머 감지 (간단한 예시 - 실제로는 더 복잡한 AI 모델 사용)"""
        # OpenCV를 사용한 간단한 움직임 감지
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 실제 구현에서는 YOLO, MediaPipe 등의 AI 모델 사용
        cv2
        return (100, 100, 200, 200)  # x, y, width, height
    
    @staticmethod
    def crop_and_track_video(input_path: str, output_path: str, job_id: str):
        """비디오 크롭 및 추적 처리"""
        try:
            # 작업 상태 업데이트
            job_status[job_id]["status"] = "processing"
            job_status[job_id]["progress"] = 10
            
            # 비디오 로드
            cap = cv2.VideoCapture(input_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 출력 비디오 설정
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (640, 480))
            
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 클라이머 감지 및 크롭
                bbox = VideoProcessor.detect_climber(frame)
                x, y, w, h = bbox
                
                # 크롭된 프레임 생성
                cropped_frame = frame[y:y+h, x:x+w]
                
                # 크기 조정
                resized_frame = cv2.resize(cropped_frame, (640, 480))
                
                # 프레임 저장
                out.write(resized_frame)
                
                # 진행률 업데이트
                frame_count += 1
                progress = int((frame_count / total_frames) * 80) + 10
                job_status[job_id]["progress"] = min(progress, 90)
                
            cap.release()
            out.release()
            
            # 작업 완료
            job_status[job_id]["status"] = "completed"
            job_status[job_id]["progress"] = 100
            job_status[job_id]["output_file"] = output_path
            
        except Exception as e:
            job_status[job_id]["status"] = "failed"
            job_status[job_id]["error"] = str(e)

@app.post("/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """비디오 파일 업로드 및 처리 시작"""
    
    # 파일 형식 검증
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="Only video files are allowed")
    
    # 고유 작업 ID 생성
    job_id = str(uuid.uuid4())[:8]
    
    # 파일 저장
    file_extension = Path(file.filename).suffix
    input_filename = f"{job_id}_input{file_extension}"
    input_path = UPLOAD_DIR / input_filename
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 출력 파일 경로
    output_filename = f"{job_id}_processed.mp4"
    output_path = PROCESSED_DIR / output_filename
    
    # 작업 상태 초기화
    job_status[job_id] = {
        "status": "processing",
        "progress": 0,
        "created_at": datetime.now().isoformat(),
        "input_file": str(input_path),
        "output_file": str(output_path)
    }
    

    # 백그라운드에서 비디오 처리 시작
    background_tasks.add_task(
        # 원래의 더미 코드
        # VideoProcessor.crop_and_track_video,
        # str(input_path),
        # str(output_path),
        # job_id
        
        cropper.process_video,
        str(input_path),
        str(output_path),
        job_status,
        job_id
    )
    return {"job_id": job_id, "message": "Video processing started"}

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in job_status:
        return {"status_code": 404, "content": {"error": f"Job ID '{job_id}' not found"}}
    return JSONResponse(content=job_status[job_id])

@app.get("/download/{job_id}")
async def download_processed_video(job_id: str):
    """처리된 비디오 다운로드"""
    
    file_path = PROCESSED_DIR / f"{job_id}_processed.mp4"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Processed file not found")

    return FileResponse(path=file_path, media_type='video/mp4', filename=f"cruxcut_processed_{job_id}.mp4")
    
    job = job_status[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    output_path = job["output_file"]
    
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Processed file not found")
    
    return FileResponse(
        path=output_path,
        media_type='video/mp4',
        filename=f"cruxcut_processed_{job_id}.mp4"
    )

@app.get("/")
async def root():
    return {"message": "CruxCut Video Processing API", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
