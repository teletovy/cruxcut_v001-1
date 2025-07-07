from fastapi import FastAPI, Request, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os
import shutil
from model import detector
import uuid

from video_cropper_oop import cropper

app = FastAPI()
templates = Jinja2Templates(directory="../templates")

# 정적 파일 서빙을 위한 디렉토리 생성 및 설정
os.makedirs("temp", exist_ok=True)
app.mount("/temp", StaticFiles(directory="temp"), name="temp")

@app.get("/")
async def home(request: Request):
    print("Home endpoint accessed")
    return templates.TemplateResponse("index.html", {"request": request})



@app.post("/process-video")
async def process_video(file: UploadFile = File(...)):
    try:
        # 임시 파일 경로 생성
        file_id = str(uuid.uuid4())
        input_path = f"temp/{file_id}_input.mp4"
        output_path = f"temp/{file_id}_output.mp4"
        
        print(f"Processing video: {input_path} -> {output_path}")
        # 업로드된 파일 저장
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 비디오 처리
        # success, frame_count, total_frames = detector.process_video(input_path, output_path)
        
        success = cropper.process_video(input_path, output_path)
        
        if success:
            return {
                "success": True,
                "file_id": file_id,
                "message": "비디오 처리가 완료되었습니다."
            }
        else:
            return {
                "success": False,
                "message": "비디오 처리 중 오류가 발생했습니다."
            }
            
    except Exception as e:
        return {"success": False, "message": str(e)}


@app.get("/download/{file_id}")
async def download_video(file_id: str):
    try:
        file_path = f"temp/{file_id}_output.mp4"
        return FileResponse(
            file_path,
            media_type="video/mp4",
            filename="detected_video.mp4"
        )
    except Exception as e:
        return {"error": str(e)}


@app.on_event("shutdown")
async def cleanup():
    # 임시 파일 정리
    shutil.rmtree("temp", ignore_errors=True)