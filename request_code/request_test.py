import cv2
import numpy as np
import time
import requests
from tkinter import Tk
from tkinter.filedialog import askopenfilename

url = "http://175.197.29.206:8080/bowling/analyze"
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)  

fourcc_h264 = cv2.VideoWriter_fourcc(*'X264')  # H.264 코덱

is_recording = False  # 녹화 여부
out = None  # VideoWriter 객체 초기화

# Tkinter 윈도우 숨기기
root = Tk()
root.withdraw()

while True:
    ret, image = cap.read()
    if not ret:
        print("카메라에서 영상을 읽을 수 없습니다.")
        break

    cv2.imshow('CAMERA', image)

    key = cv2.waitKey(30) & 0xFF

    if key == ord('p'):
        filename = f"frame_{int(time.time())}.jpg"  # 현재 시간 기반 파일명
        cv2.imwrite(filename, image)
        print(f"프레임 저장: {filename}")
        files = {"file": open(filename, "rb")}
        response = requests.post(url, files=files)

        print("응답 상태 코드:", response.status_code)
        print("응답 데이터:", response.text)

    elif key == ord('w'):  # 'w' 키를 눌러 파일 선택 대화상자 열기
        filepath = askopenfilename(title="파일 선택", filetypes=[("All Files", "*.*")])
        if filepath:  # 파일이 선택되면
            with open(filepath, "rb") as file:
                files = {"file": file}
                response = requests.post(url, files=files)

            print(f"파일 {filepath} 전송 완료")
            print("응답 상태 코드:", response.status_code)
            print("응답 데이터:", response.text)

    elif key == ord('v'):  # 'v' 키를 눌러 비디오 촬영 시작/종료
        if not is_recording:  # 녹화 시작
            filename = f"video_{int(time.time())}.mp4"  # mp4 형식으로 파일명 생성
            # XVID 코덱을 사용하는 다른 비디오 포맷으로 저장 (mp4 코덱 호환성 문제 해결)
            out = cv2.VideoWriter(filename, fourcc_h264, 20.0, (640, 480))  # 640x480으로 설정
            is_recording = True
            print("녹화를 시작합니다.")
        else:  # 녹화 종료
            is_recording = False
            out.release()  # 비디오 파일을 종료
            print("녹화를 종료합니다.")
            # 녹화 종료 후 동영상 파일 전송
            with open(filename, "rb") as video_file:
                files = {"file": video_file}
                response = requests.post(url, files=files)

            print(f"동영상 파일 {filename} 전송 완료")
            print("응답 상태 코드:", response.status_code)
            print("응답 데이터:", response.text)

    elif key == ord('q'):  # 'q' 키를 눌러 종료
        print("종료합니다.")
        break

    # 녹화 중일 때 비디오 파일에 프레임 쓰기
    if is_recording:
        out.write(image)

cap.release()
cv2.destroyAllWindows()