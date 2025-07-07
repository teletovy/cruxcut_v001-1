# FastAPI 프로젝트 설정 스크립트
import subprocess
import sys
import os

def install_packages():
    """필요한 패키지들을 설치합니다."""
    packages = [
        "fastapi",
        "uvicorn[standard]",
        "python-multipart",
        "opencv-python",
        "moviepy",
        "pillow",
        "numpy",
        "python-jose[cryptography]",
        "passlib[bcrypt]",
        "aiofiles"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def create_directories():
    """필요한 디렉토리들을 생성합니다."""
    directories = [
        "uploads",
        "processed",
        "temp"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

if __name__ == "__main__":
    print("Setting up FastAPI video processing environment...")
    install_packages()
    create_directories()
    print("Setup complete!")
