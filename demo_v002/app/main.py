from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import shutil
import uuid
import cv2
from PIL import Image
import io
import base64

# Pydantic models for request bodies
class BoundingBoxRequest(BaseModel):
    file_id: str
    bbox: dict  # {"x": int, "y": int, "width": int, "height": int}

app = FastAPI()

# 정적 파일 서빙을 위한 디렉토리 생성 및 설정
os.makedirs("temp", exist_ok=True)
app.mount("/temp", StaticFiles(directory="temp"), name="temp")

@app.get("/")
async def home():
    """메인 페이지를 반환합니다."""
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>O-CAE-Auto-Camera-Editing-Service-for-Climbers</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
        }
        .container {
            text-align: center;
        }
        .step {
            margin: 30px 0;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 10px;
            display: none;
        }
        .step.active {
            display: block;
        }
        .step h3 {
            margin-top: 0;
            color: #333;
        }
        .progress-bar {
            width: 100%;
            height: 20px;
            background-color: #f0f0f0;
            border-radius: 10px;
            margin: 20px 0;
            display: none;
        }
        .progress {
            width: 0%;
            height: 100%;
            background-color: #4CAF50;
            border-radius: 10px;
            transition: width 0.3s ease;
        }
        .btn {
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            margin: 5px;
        }
        .btn:hover {
            background-color: #45a049;
        }
        .btn:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        #status {
            margin: 20px 0;
            font-weight: bold;
        }
        .canvas-container {
            position: relative;
            display: inline-block;
            margin: 20px 0;
        }
        #frameCanvas {
            border: 2px solid #ddd;
            cursor: crosshair;
            max-width: 100%;
            height: auto;
        }
        .bounding-box {
            position: absolute;
            border: 2px solid #ff0000;
            background-color: rgba(255, 0, 0, 0.1);
            pointer-events: none;
        }
        .coordinates-info {
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            text-align: left;
        }
        .file-input-wrapper {
            position: relative;
            display: inline-block;
            cursor: pointer;
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            margin: 10px;
            transition: background-color 0.3s ease;
        }
        .file-input-wrapper:hover {
            background-color: #45a049;
        }
        .file-input-wrapper input[type=file] {
            position: absolute;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            opacity: 0;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>O-CAE: Auto-Camera-Editing-Service-for-Climbers</h1>
        <p>30초 이하의 클라이밍 영상을 업로드하여 편집 영역을 선택해주세요.</p>
        
        <!-- Step 1: Video Upload -->
        <div class="step active" id="step1">
            <h3>1단계: 비디오 업로드</h3>
            <div class="file-input-wrapper">
                <input type="file" id="videoInput" accept="video/mp4,video/avi,video/mov">
                비디오 선택하기
            </div>
            <div id="status"></div>
            <div class="progress-bar" id="progressBar">
                <div class="progress" id="progress"></div>
            </div>
        </div>

        <!-- Step 2: Bounding Box Selection -->
        <div class="step" id="step2">
            <h3>2단계: 편집 영역 선택</h3>
            <p>첫 번째 프레임에서 편집하고 싶은 영역을 마우스로 드래그하여 선택해주세요.</p>
            <div class="canvas-container">
                <canvas id="frameCanvas"></canvas>
                <div id="boundingBox" class="bounding-box"></div>
            </div>
            <div class="coordinates-info" id="coordinatesInfo">
                <strong>선택된 좌표:</strong><br>
                시작점: (<span id="startX">-</span>, <span id="startY">-</span>)<br>
                끝점: (<span id="endX">-</span>, <span id="endY">-</span>)<br>
                크기: <span id="boxWidth">-</span> x <span id="boxHeight">-</span>
            </div>
            <button class="btn" id="resetBtn">선택 초기화</button>
            <button class="btn" id="processBtn" disabled>편집 시작</button>
        </div>

        <!-- Step 3: Processing -->
        <div class="step" id="step3">
            <h3>3단계: 비디오 처리 중...</h3>
            <div class="progress-bar" style="display: block;">
                <div class="progress" id="processProgress"></div>
            </div>
            <div id="processStatus">비디오를 처리하고 있습니다...</div>
        </div>

        <!-- Step 4: Download -->
        <div class="step" id="step4">
            <h3>4단계: 완료</h3>
            <p>편집이 완료되었습니다!</p>
            <button class="btn" id="downloadBtn">결과 비디오 다운로드</button>
            <button class="btn" id="newVideoBtn">새 비디오 업로드</button>
        </div>
    </div>

    <script>
        const videoInput = document.getElementById('videoInput');
        const status = document.getElementById('status');
        const progressBar = document.getElementById('progressBar');
        const progress = document.getElementById('progress');
        const frameCanvas = document.getElementById('frameCanvas');
        const ctx = frameCanvas.getContext('2d');
        const boundingBox = document.getElementById('boundingBox');
        const coordinatesInfo = document.getElementById('coordinatesInfo');
        const resetBtn = document.getElementById('resetBtn');
        const processBtn = document.getElementById('processBtn');
        const downloadBtn = document.getElementById('downloadBtn');
        const newVideoBtn = document.getElementById('newVideoBtn');
        
        let currentFileId = null;
        let isDrawing = false;
        let startX, startY, endX, endY;
        let canvasRect;

        // Step management
        function showStep(stepNumber) {
            document.querySelectorAll('.step').forEach(step => step.classList.remove('active'));
            document.getElementById(`step${stepNumber}`).classList.add('active');
        }

        // Video upload handler
        videoInput.addEventListener('change', async (e) => {
            console.log('File input changed:', e.target.files);
            const file = e.target.files[0];
            if (!file) {
                console.log('No file selected');
                return;
            }

            console.log('Selected file:', file.name, file.type, file.size);
            status.textContent = "비디오 업로드 중...";
            progressBar.style.display = 'block';
            progress.style.width = '0%';

            const formData = new FormData();
            formData.append('file', file);

            try {
                console.log('Sending request to /upload-video');
                const response = await fetch('/upload-video', {
                    method: 'POST',
                    body: formData
                });
                
                console.log('Response status:', response.status);
                const data = await response.json();
                console.log('Response data:', data);
                
                if (data.success) {
                    currentFileId = data.file_id;
                    status.textContent = "첫 프레임을 불러오는 중...";
                    progress.style.width = '50%';
                    
                    // Load first frame
                    await loadFirstFrame(data.file_id);
                    
                    progress.style.width = '100%';
                    status.textContent = "업로드 완료!";
                    
                    setTimeout(() => {
                        showStep(2);
                    }, 1000);
                } else {
                    status.textContent = data.message || '업로드 실패';
                }
                
            } catch (error) {
                console.error('Upload error:', error);
                status.textContent = "업로드 중 오류가 발생했습니다: " + error.message;
            }
        });

        // Load first frame
        async function loadFirstFrame(fileId) {
            try {
                console.log('Loading first frame for file ID:', fileId);
                const response = await fetch(`/get-first-frame/${fileId}`);
                console.log('First frame response status:', response.status);
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                const blob = await response.blob();
                console.log('First frame blob size:', blob.size);
                
                const img = new Image();
                
                img.onload = function() {
                    console.log('Image loaded:', img.width, 'x', img.height);
                    // Set canvas size to match image
                    frameCanvas.width = img.width;
                    frameCanvas.height = img.height;
                    
                    // Draw image on canvas
                    ctx.drawImage(img, 0, 0);
                    
                    // Update canvas rect for coordinate calculation
                    canvasRect = frameCanvas.getBoundingClientRect();
                    console.log('Canvas rect:', canvasRect);
                };
                
                img.onerror = function() {
                    console.error('Failed to load image');
                    status.textContent = "이미지 로드 실패";
                };
                
                img.src = URL.createObjectURL(blob);
            } catch (error) {
                console.error('Error loading first frame:', error);
                status.textContent = "첫 프레임 로드 중 오류: " + error.message;
            }
        }

        // Canvas mouse events for bounding box
        frameCanvas.addEventListener('mousedown', (e) => {
            isDrawing = true;
            canvasRect = frameCanvas.getBoundingClientRect();
            
            const scaleX = frameCanvas.width / canvasRect.width;
            const scaleY = frameCanvas.height / canvasRect.height;
            
            startX = (e.clientX - canvasRect.left) * scaleX;
            startY = (e.clientY - canvasRect.top) * scaleY;
            
            boundingBox.style.display = 'none';
        });

        frameCanvas.addEventListener('mousemove', (e) => {
            if (!isDrawing) return;
            
            const scaleX = frameCanvas.width / canvasRect.width;
            const scaleY = frameCanvas.height / canvasRect.height;
            
            endX = (e.clientX - canvasRect.left) * scaleX;
            endY = (e.clientY - canvasRect.top) * scaleY;
            
            updateBoundingBox();
        });

        frameCanvas.addEventListener('mouseup', (e) => {
            if (!isDrawing) return;
            
            isDrawing = false;
            const scaleX = frameCanvas.width / canvasRect.width;
            const scaleY = frameCanvas.height / canvasRect.height;
            
            endX = (e.clientX - canvasRect.left) * scaleX;
            endY = (e.clientY - canvasRect.top) * scaleY;
            
            updateBoundingBox();
            updateCoordinatesInfo();
            processBtn.disabled = false;
        });

        function updateBoundingBox() {
            const left = Math.min(startX, endX);
            const top = Math.min(startY, endY);
            const width = Math.abs(endX - startX);
            const height = Math.abs(endY - startY);
            
            const scaleX = canvasRect.width / frameCanvas.width;
            const scaleY = canvasRect.height / frameCanvas.height;
            
            boundingBox.style.display = 'block';
            boundingBox.style.left = (left * scaleX) + 'px';
            boundingBox.style.top = (top * scaleY) + 'px';
            boundingBox.style.width = (width * scaleX) + 'px';
            boundingBox.style.height = (height * scaleY) + 'px';
        }

        function updateCoordinatesInfo() {
            const left = Math.min(startX, endX);
            const top = Math.min(startY, endY);
            const width = Math.abs(endX - startX);
            const height = Math.abs(endY - startY);
            
            document.getElementById('startX').textContent = Math.round(left);
            document.getElementById('startY').textContent = Math.round(top);
            document.getElementById('endX').textContent = Math.round(left + width);
            document.getElementById('endY').textContent = Math.round(top + height);
            document.getElementById('boxWidth').textContent = Math.round(width);
            document.getElementById('boxHeight').textContent = Math.round(height);
        }

        // Reset bounding box
        resetBtn.addEventListener('click', () => {
            boundingBox.style.display = 'none';
            document.getElementById('startX').textContent = '-';
            document.getElementById('startY').textContent = '-';
            document.getElementById('endX').textContent = '-';
            document.getElementById('endY').textContent = '-';
            document.getElementById('boxWidth').textContent = '-';
            document.getElementById('boxHeight').textContent = '-';
            processBtn.disabled = true;
        });

        // Process video with bounding box
        processBtn.addEventListener('click', async () => {
            if (!currentFileId) return;
            
            const left = Math.min(startX, endX);
            const top = Math.min(startY, endY);
            const width = Math.abs(endX - startX);
            const height = Math.abs(endY - startY);
            
            showStep(3);
            
            const coordinates = {
                x: Math.round(left),
                y: Math.round(top),
                width: Math.round(width),
                height: Math.round(height)
            };
            
            try {
                const response = await fetch('/process-with-bbox', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        file_id: currentFileId,
                        bbox: coordinates
                    })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    document.getElementById('processProgress').style.width = '100%';
                    document.getElementById('processStatus').textContent = '처리 완료!';
                    setTimeout(() => {
                        showStep(4);
                    }, 1000);
                } else {
                    document.getElementById('processStatus').textContent = '처리 중 오류가 발생했습니다: ' + data.message;
                }
                
            } catch (error) {
                console.error('Error:', error);
                document.getElementById('processStatus').textContent = '처리 중 오류가 발생했습니다.';
            }
        });

        // Download processed video
        downloadBtn.addEventListener('click', () => {
            if (currentFileId) {
                window.location.href = `/download/${currentFileId}`;
            }
        });

        // Start new video
        newVideoBtn.addEventListener('click', () => {
            currentFileId = null;
            videoInput.value = '';
            resetBtn.click();
            showStep(1);
        });
    </script>
</body>
</html>"""
    return HTMLResponse(content=html_content)

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """비디오 파일을 업로드하고 첫 프레임을 추출합니다."""
    try:
        # 임시 파일 경로 생성
        file_id = str(uuid.uuid4())
        input_path = f"temp/{file_id}_input.mp4"
        
        print(f"Uploading video: {input_path}")
        
        # 업로드된 파일 저장
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 첫 프레임 추출
        success = extract_first_frame(input_path, file_id)
        
        if success:
            return {
                "success": True,
                "file_id": file_id,
                "message": "비디오 업로드 및 첫 프레임 추출이 완료되었습니다."
            }
        else:
            return {
                "success": False,
                "message": "첫 프레임 추출 중 오류가 발생했습니다."
            }
            
    except Exception as e:
        print(f"Upload error: {str(e)}")
        return {"success": False, "message": str(e)}

def extract_first_frame(video_path: str, file_id: str) -> bool:
    """비디오에서 첫 번째 프레임을 추출합니다."""
    try:
        # OpenCV로 비디오 열기
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return False
        
        # 첫 번째 프레임 읽기
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("Error: Could not read first frame")
            return False
        
        # 프레임을 이미지 파일로 저장
        frame_path = f"temp/{file_id}_frame.jpg"
        cv2.imwrite(frame_path, frame)
        
        print(f"First frame extracted: {frame_path}")
        return True
        
    except Exception as e:
        print(f"Frame extraction error: {str(e)}")
        return False

@app.get("/get-first-frame/{file_id}")
async def get_first_frame(file_id: str):
    """추출된 첫 프레임 이미지를 반환합니다."""
    try:
        frame_path = f"temp/{file_id}_frame.jpg"
        
        if not os.path.exists(frame_path):
            raise HTTPException(status_code=404, detail="Frame not found")
        
        return FileResponse(
            frame_path,
            media_type="image/jpeg",
            filename="first_frame.jpg"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-with-bbox")
async def process_with_bbox(request: BoundingBoxRequest):
    """바운딩 박스 좌표를 받아서 비디오를 처리합니다."""
    try:
        file_id = request.file_id
        bbox = request.bbox
        
        input_path = f"temp/{file_id}_input.mp4"
        output_path = f"temp/{file_id}_output.mp4"
        
        print(f"Processing video with bbox: {bbox}")
        print(f"Input: {input_path} -> Output: {output_path}")
        
        if not os.path.exists(input_path):
            return {
                "success": False,
                "message": "원본 비디오 파일을 찾을 수 없습니다."
            }
        
        # 바운딩 박스를 이용한 비디오 처리
        success = process_video_with_crop(input_path, output_path, bbox)
        
        if success:
            return {
                "success": True,
                "message": "비디오 처리가 완료되었습니다.",
                "bbox_used": bbox
            }
        else:
            return {
                "success": False,
                "message": "비디오 처리 중 오류가 발생했습니다."
            }
            
    except Exception as e:
        print(f"Processing error: {str(e)}")
        return {"success": False, "message": str(e)}

def process_video_with_crop(input_path: str, output_path: str, bbox: dict) -> bool:
    """바운딩 박스를 이용해 비디오를 크롭합니다."""
    try:
        # OpenCV로 비디오 열기
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {input_path}")
            return False
        
        # 비디오 속성 가져오기
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Original video: {width}x{height} @ {fps}fps")
        
        # 바운딩 박스 좌표
        x = max(0, bbox["x"])
        y = max(0, bbox["y"])
        w = min(bbox["width"], width - x)
        h = min(bbox["height"], height - y)
        
        print(f"Crop region: ({x}, {y}) {w}x{h}")
        
        # 비디오 라이터 설정
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 프레임 크롭
            cropped_frame = frame[y:y+h, x:x+w]
            
            # 크롭된 프레임 저장
            out.write(cropped_frame)
            frame_count += 1
        
        cap.release()
        out.release()
        
        print(f"Processed {frame_count} frames")
        return True
        
    except Exception as e:
        print(f"Video processing error: {str(e)}")
        return False

@app.get("/download/{file_id}")
async def download_video(file_id: str):
    """처리된 비디오를 다운로드합니다."""
    try:
        file_path = f"temp/{file_id}_output.mp4"
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Processed video not found")
        
        return FileResponse(
            file_path,
            media_type="video/mp4",
            filename="edited_video.mp4"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("shutdown")
async def cleanup():
    """애플리케이션 종료 시 임시 파일 정리"""
    shutil.rmtree("temp", ignore_errors=True)

# 개발용 엔드포인트 - 현재 임시 파일들 확인
@app.get("/debug/temp-files")
async def list_temp_files():
    """디버깅용: 임시 파일 목록 확인"""
    try:
        files = os.listdir("temp")
        return {"temp_files": files}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)