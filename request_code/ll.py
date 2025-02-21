import cv2
import numpy as np
import sys
import time
import requests
from tkinter import Tk
from tkinter.filedialog import askopenfilename

url = "http://172.30.1.98:8080/test/cam"
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

is_recording = False  # 녹화 여부
out = None  # VideoWriter 객체 초기화
start_time = None  # 녹화 시작 시간

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

    elif key == ord('q'):  # 'q' 키를 눌러 종료
        print("종료합니다.")
        break

cap.release()
cv2.destroyAllWindows()
