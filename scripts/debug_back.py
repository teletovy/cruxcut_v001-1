import requests
import json
import time
import base64
from pathlib import Path
import cv2
import numpy as np
import os

class VideoAPITester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.job_id = None
        
    def create_test_video(self, output_path="test_video.mp4", duration=5, fps=30):
        """테스트용 비디오 생성"""
        print(f"🎬 Creating test video: {output_path}")
        
        # 비디오 설정
        width, height = 640, 480
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        total_frames = duration * fps
        
        for i in range(total_frames):
            # 간단한 애니메이션 프레임 생성
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # 배경색 변화
            bg_color = int(i * 255 / total_frames)
            frame[:] = (bg_color // 3, bg_color // 2, bg_color)
            
            # 움직이는 원 그리기
            center_x = int(width * (0.2 + 0.6 * i / total_frames))
            center_y = height // 2
            radius = 30 + int(20 * np.sin(i * 0.1))
            
            cv2.circle(frame, (center_x, center_y), radius, (0, 255, 255), -1)
            
            # 프레임 번호 표시
            cv2.putText(frame, f"Frame {i+1}/{total_frames}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        print(f"✅ Test video created: {output_path}")
        return output_path
    
    def test_server_status(self):
        """서버 상태 확인"""
        print("\n🔍 Testing server status...")
        try:
            response = requests.get(f"{self.base_url}/")
            if response.status_code == 200:
                print("✅ Server is running")
                print(f"Response: {response.json()}")
                return True
            else:
                print(f"❌ Server error: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to server. Make sure it's running on http://localhost:8000")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def test_video_upload(self, video_path):
        """비디오 업로드 테스트"""
        print(f"\n📤 Testing video upload: {video_path}")
        
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            return False
        
        try:
            with open(video_path, 'rb') as f:
                files = {'file': (video_path, f, 'video/mp4')}
                response = requests.post(f"{self.base_url}/upload", files=files)
            
            if response.status_code == 200:
                result = response.json()
                self.job_id = result['job_id']
                print(f"✅ Upload successful!")
                print(f"Job ID: {self.job_id}")
                print(f"Video Info: {result['video_info']}")
                
                # 첫 프레임 저장
                if 'first_frame' in result:
                    self.save_base64_image(result['first_frame'], f"first_frame_{self.job_id}.jpg")
                    print(f"🖼️ First frame saved as: first_frame_{self.job_id}.jpg")
                
                return True
            else:
                print(f"❌ Upload failed: {response.status_code}")
                print(f"Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Upload error: {e}")
            return False
    
    def save_base64_image(self, base64_string, filename):
        """Base64 이미지를 파일로 저장"""
        try:
            # "data:image/jpeg;base64," 부분 제거
            if base64_string.startswith('data:image'):
                base64_string = base64_string.split(',')[1]
            
            image_data = base64.b64decode(base64_string)
            with open(filename, 'wb') as f:
                f.write(image_data)
            return True
        except Exception as e:
            print(f"❌ Failed to save image: {e}")
            return False
    
    def test_status_check(self):
        """작업 상태 확인 테스트"""
        if not self.job_id:
            print("❌ No job ID available")
            return False
        
        print(f"\n📊 Testing status check for job: {self.job_id}")
        
        try:
            response = requests.get(f"{self.base_url}/status/{self.job_id}")
            if response.status_code == 200:
                status = response.json()
                print(f"✅ Status check successful!")
                print(f"Status: {json.dumps(status, indent=2)}")
                return True
            else:
                print(f"❌ Status check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Status check error: {e}")
            return False
    
    def test_video_processing(self, bbox=None):
        """비디오 처리 테스트"""
        if not self.job_id:
            print("❌ No job ID available")
            return False
        
        if bbox is None:
            bbox = {"x": 100, "y": 100, "width": 300, "height": 200}
        
        print(f"\n⚙️ Testing video processing with bounding box: {bbox}")
        
        try:
            payload = {
                "job_id": self.job_id,
                "bounding_box": bbox,
                "additional_params": {}
            }
            
            response = requests.post(
                f"{self.base_url}/process",
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Processing started!")
                print(f"Result: {json.dumps(result, indent=2)}")
                return True
            else:
                print(f"❌ Processing failed: {response.status_code}")
                print(f"Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Processing error: {e}")
            return False
    
    def monitor_processing(self, check_interval=2, max_wait=60):
        """처리 진행 상황 모니터링"""
        if not self.job_id:
            print("❌ No job ID available")
            return False
        
        print(f"\n👁️ Monitoring processing progress...")
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                response = requests.get(f"{self.base_url}/status/{self.job_id}")
                if response.status_code == 200:
                    status = response.json()
                    current_status = status.get('status', 'unknown')
                    progress = status.get('progress', 0)
                    
                    print(f"⏳ Status: {current_status}, Progress: {progress}%")
                    
                    if current_status == 'completed':
                        print("✅ Processing completed!")
                        return True
                    elif current_status == 'failed':
                        print(f"❌ Processing failed: {status.get('error', 'Unknown error')}")
                        return False
                    
                    time.sleep(check_interval)
                else:
                    print(f"❌ Status check failed: {response.status_code}")
                    return False
                    
            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                return False
        
        print(f"⏰ Timeout after {max_wait} seconds")
        return False
    
    def test_download(self):
        """처리된 비디오 다운로드 테스트"""
        if not self.job_id:
            print("❌ No job ID available")
            return False
        
        print(f"\n⬇️ Testing video download...")
        
        try:
            response = requests.get(f"{self.base_url}/download/{self.job_id}")
            
            if response.status_code == 200:
                output_filename = f"downloaded_{self.job_id}.mp4"
                with open(output_filename, 'wb') as f:
                    f.write(response.content)
                
                print(f"✅ Download successful!")
                print(f"Saved as: {output_filename}")
                print(f"File size: {len(response.content)} bytes")
                return True
            else:
                print(f"❌ Download failed: {response.status_code}")
                print(f"Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Download error: {e}")
            return False
    
    def test_cleanup(self):
        """파일 정리 테스트"""
        if not self.job_id:
            print("❌ No job ID available")
            return False
        
        print(f"\n🧹 Testing cleanup...")
        
        try:
            response = requests.delete(f"{self.base_url}/cleanup/{self.job_id}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Cleanup successful!")
                print(f"Result: {json.dumps(result, indent=2)}")
                return True
            else:
                print(f"❌ Cleanup failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Cleanup error: {e}")
            return False
    
    def run_full_test(self):
        """전체 테스트 실행"""
        print("🚀 Starting full API test...")
        
        # 1. 서버 상태 확인
        if not self.test_server_status():
            return False
        
        # 2. 테스트 비디오 생성
        test_video = self.create_test_video()
        
        # 3. 비디오 업로드
        if not self.test_video_upload(test_video):
            return False
        
        # 4. 상태 확인
        if not self.test_status_check():
            return False
        
        # 5. 비디오 처리 시작
        if not self.test_video_processing():
            return False
        
        # 6. 처리 진행 상황 모니터링
        if not self.monitor_processing():
            return False
        
        # 7. 처리된 비디오 다운로드
        if not self.test_download():
            return False
        
        # 8. 파일 정리
        if not self.test_cleanup():
            return False
        
        # 9. 테스트 파일 정리
        try:
            os.remove(test_video)
            print(f"🧹 Cleaned up test video: {test_video}")
        except:
            pass
        
        print("\n🎉 All tests completed successfully!")
        return True

def main():
    print("=" * 60)
    print("🔧 FastAPI Video Processing API Debug Tester")
    print("=" * 60)
    
    tester = VideoAPITester()
    
    # 개별 테스트 실행 옵션
    print("\nSelect test option:")
    print("1. Run full test")
    print("2. Test server status only")
    print("3. Test with existing video file")
    
    choice = input("\nEnter choice (1-3, default=1): ").strip() or "1"
    
    if choice == "1":
        tester.run_full_test()
    elif choice == "2":
        tester.test_server_status()
    elif choice == "3":
        video_path = input("Enter video file path: ").strip()
        if video_path and os.path.exists(video_path):
            tester.test_video_upload(video_path)
            tester.test_status_check()
        else:
            print("❌ Video file not found")
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()