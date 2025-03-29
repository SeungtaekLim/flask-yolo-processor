import cv2
import time
from ultralytics import YOLO

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

# YOLO 모델 로드 (pose 모델)
model = YOLO("yolo11m-pose.pt")  # pose 추정 모델을 로드

# 키포인트 이름
KEYPOINT_NAMES = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear", 
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow", 
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip", 
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

# 연속 조건 체크를 위한 변수
wrist_condition_met_count = 0
recording_started = False
is_recording = False
video_writer = None
start_time = None  # 녹화 시작 시간을 추적하기 위한 변수

# 실시간 처리 루프
while cap.isOpened():
    while(1):
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
    
        # YOLO 모델을 사용하여 현재 프레임에서 키포인트 추출
        results = model(frame)
        keypoints = results[0].keypoints  # keypoints를 얻기
        xy = keypoints.xy  # (x, y) 좌표
    
        # keypoints가 0 (None)을 포함하지 않도록 필터링
        if len(xy[0]) < 16:
            continue
    
        # Right Wrist와 Left Wrist의 y 좌표 가져오기
        right_wrist = xy[0][9]  # Right Wrist의 (x, y)
        left_wrist = xy[0][10]  # Left Wrist의 (x, y)
        right_ear = xy[0][3]  # Right Ear의 (x, y)
        left_ear = xy[0][2]  # Left Ear의 (x, y)
    
        wrist_condition_met = False
        
        # 귀가 탐지된 경우에만 체크
        if (right_ear[0] != 0 and right_ear[1] != 0) or (left_ear[0] != 0 and left_ear[1] != 0):
            print("귀가 감지되었습니다.")  # 귀가 감지된 경우
            
            # 오른쪽 손목이 존재하고, 귀보다 높은지 확인
            if right_wrist[0] != 0 and right_wrist[1] != 0:
                # 오른쪽 손목이 귀보다 높은지 확인
                print("오른손이 감지되었습니다.")
                if right_wrist[1] < right_ear[1] or right_wrist[1] < left_ear[1]:
                    wrist_condition_met = True
        
            # 왼쪽 손목이 존재하고, 귀보다 높은지 확인
            if left_wrist[0] != 0 and left_wrist[1] != 0:
                # 왼쪽 손목이 귀보다 높은지 확인
                print("왼손이 감지되었습니다.")
                if left_wrist[1] < right_ear[1] or left_wrist[1] < left_ear[1]:
                    wrist_condition_met = True
        else:
            print("귀가 감지되지 않았습니다.")  # 귀가 감지되지 않은 경우
    
        # 연속해서 조건이 충족되었는지 확인
        if wrist_condition_met:
            wrist_condition_met_count += 1
        else:
            wrist_condition_met_count = 0  # 조건이 충족되지 않으면 카운트를 리셋
        
        print(wrist_condition_met_count)
    
        # 2번 연속으로 wrist_condition_met
        if wrist_condition_met_count == 2:
            break
    
        # 1초 대기 후 다음 프레임을 캡처
        time.sleep(1)
    
        # YOLO 결과로 주석이 달린 이미지를 생성 (numpy.ndarray 반환)
        annotated_image = results[0].plot()  # 결과 이미지를 얻음
        
        # OpenCV로 이미지를 출력
        cv2.imshow("Annotated Image", annotated_image)
        
        # 'q' 키를 누르면 루프 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    while(1):
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # 비디오 녹화 시작
        if not is_recording:  # 녹화 시작
            filename = f"video_{int(time.time())}.mp4"  # mp4 형식으로 파일명 생성
            fourcc_h264 = cv2.VideoWriter_fourcc(*'H264')
            video_writer = cv2.VideoWriter(filename, fourcc_h264, 20.0, (640, 480))  # 640x480으로 설정
            start_time = time.time()  # 녹화 시작 시간
            is_recording = True
            print("녹화를 시작합니다.")

        # 녹화 중인 경우, 프레임을 파일에 기록
        video_writer.write(frame)
        
        # 8초가 지나면 녹화 종료
        if time.time() - start_time >= 8:
            is_recording = False
            video_writer.release()  # 비디오 파일을 저장하고 리소스를 해제
            print("녹화가 종료되었습니다.")
            break
        
        cv2.imshow('CAMERA', frame)
        
        # 'q' 키를 누르면 루프 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 종료 시 웹캠과 비디오 캡처 객체를 릴리즈
cap.release()
cv2.destroyAllWindows()
