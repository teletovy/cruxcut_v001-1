import cv2

print(f"실제 사용되는 OpenCV 버전: {cv2.__version__}")

# contrib 기능이 있는지 확인
try:
    # contrib에만 있는 기능 테스트
    cv2.xfeatures2d.SIFT_create()
    print("OpenCV-contrib 기능 사용 가능")
except AttributeError:
    print("OpenCV-contrib 기능 없음 (일반 opencv-python)")
except Exception as e:
    print(f"테스트 중 오류: {e}")