import cv2
from ultralytics import YOLO
import numpy as np

# 동영상 파일 경로
video_file = 'cam.mp4'

# YOLO 모델 로드 (여기서는 자세 추정 모델을 로드한다고 가정)
model = YOLO("yolo11m-pose.pt")  # pose 추정 모델을 로드

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(video_file)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 화면 크기 설정 (동영상의 해상도에 맞추기)
WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 동영상 저장을 위한 설정 (OpenCV 사용)
fourcc_h264 = cv2.VideoWriter_fourcc(*'X264')
output_video = cv2.VideoWriter('animation.mp4', fourcc_h264, 20, (WIDTH, HEIGHT))  # 초당 20프레임으로 저장

# 키포인트 크기 (원 크기)
KEYPOINT_RADIUS = 5

# 키포인트 이름 (순서대로)
KEYPOINT_NAMES = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear", 
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow", 
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip", 
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

# 배열을 통한 키포인트 데이터 저장
keypoints_data = []

# 동영상에서 프레임 하나씩 읽어오기
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 프레임을 모델에 넣어 예측 수행
    results = model(frame)  # 프레임을 모델에 넣어 예측
    
    # keypoints 추출
    keypoints = results[0].keypoints  # keypoints를 얻기
    
    # keypoints.xy: (x, y) 좌표, keypoints.conf: 각 keypoint의 confidence
    xy = keypoints.xy  # 픽셀 좌표
    conf = keypoints.conf  # 신뢰도

    # 각 키포인트의 좌표를 배열에 저장
    frame_keypoints = []
    for i in range(len(xy[0])):  # 각 사람마다 반복
        x, y = xy[0][i]  # (x, y) 좌표
        frame_keypoints.append((x, y))
    
    keypoints_data.append(frame_keypoints)  # 프레임 키포인트 데이터 추가

def mk_keypoints_video():
    current_frame = 0
    while current_frame < len(keypoints_data):
        # 현재 프레임에 해당하는 키포인트 좌표 가져오기
        keypoints = keypoints_data[current_frame]
        
        # 흰 배경 이미지 생성 (크기: (WIDTH, HEIGHT))
        frame = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255  # 흰 배경
        
        # 키포인트를 그리기 전에 이름과 좌표를 화면에 출력
        y_offset = 30  # 이름 출력 시 y좌표 오프셋, 처음에는 30으로 설정
        
        # KEYPOINT_NAMES 만큼 출력하지 않고 항상 모든 키포인트를 출력
        for i in range(17):  # 17개 키포인트 전부를 출력
            x, y = keypoints[i]  # 현재 키포인트의 좌표
            
            # 키포인트가 (0, 0)인 경우 "None"을 출력하고 그렇지 않으면 (x, y) 값을 출력
            if x == 0 and y == 0:
                coordinates_text = "None"
            else:
                coordinates_text = f"({int(x)}, {int(y)})"
            
            # 키포인트를 빨간색 원으로 그리기
            if x != 0 and y != 0:
                cv2.circle(frame, (int(x), int(y)), KEYPOINT_RADIUS, (0, 0, 255), -1)

            # 키포인트 이름과 좌표를 출력
            text = f"{KEYPOINT_NAMES[i]}: {coordinates_text}"
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            y_offset += 20  # 각 텍스트의 y 오프셋을 증가시켜서 겹치지 않도록 함
        
        # draw_pose_connections 호출하여 선을 그리기
        draw_pose_connections(keypoints, frame)

        # 그려진 프레임을 동영상에 추가
        output_video.write(frame)

        # 다음 프레임으로 이동
        current_frame += 1

    # 동영상 저장 종료
    cap.release()
    output_video.release()

def draw_pose_connections(keypoint_positions, frame):
    # 선을 그릴 때 사용할 색상 (BGR 포맷)
    GREEN = (0, 255, 0)

    # 각 키포인트가 유효한지 체크하는 함수
    def is_valid_keypoint(idx):
        return keypoint_positions[idx][0] != 0 and keypoint_positions[idx][1] != 0

    # 왼쪽 팔꿈치와 왼쪽 어깨 연결
    if len(keypoint_positions) > 7:
        if is_valid_keypoint(7) and is_valid_keypoint(5):
            cv2.line(frame, tuple(map(int, keypoint_positions[7])), tuple(map(int, keypoint_positions[5])), GREEN, 2)

    # 왼쪽 팔꿈치와 왼쪽 손목 연결
    if len(keypoint_positions) > 9:
        if is_valid_keypoint(7) and is_valid_keypoint(9):
            cv2.line(frame, tuple(map(int, keypoint_positions[7])), tuple(map(int, keypoint_positions[9])), GREEN, 2)

    # 오른쪽 팔꿈치와 오른쪽 어깨 연결
    if len(keypoint_positions) > 8:
        if is_valid_keypoint(8) and is_valid_keypoint(6):
            cv2.line(frame, tuple(map(int, keypoint_positions[8])), tuple(map(int, keypoint_positions[6])), GREEN, 2)

    # 오른쪽 팔꿈치와 오른쪽 손목 연결
    if len(keypoint_positions) > 10:
        if is_valid_keypoint(8) and is_valid_keypoint(10):
            cv2.line(frame, tuple(map(int, keypoint_positions[8])), tuple(map(int, keypoint_positions[10])), GREEN, 2)

    # 왼쪽 무릎과 왼쪽 발목 연결
    if len(keypoint_positions) > 15:
        if is_valid_keypoint(13) and is_valid_keypoint(15):
            cv2.line(frame, tuple(map(int, keypoint_positions[13])), tuple(map(int, keypoint_positions[15])), GREEN, 2)

    # 오른쪽 무릎과 오른쪽 발목 연결
    if len(keypoint_positions) > 16:
        if is_valid_keypoint(14) and is_valid_keypoint(16):
            cv2.line(frame, tuple(map(int, keypoint_positions[14])), tuple(map(int, keypoint_positions[16])), GREEN, 2)

    # 몸통 선 (왼쪽 어깨 - 오른쪽 어깨)
    if len(keypoint_positions) > 6:
        if is_valid_keypoint(5) and is_valid_keypoint(6):
            cv2.line(frame, tuple(map(int, keypoint_positions[5])), tuple(map(int, keypoint_positions[6])), GREEN, 2)

    # 왼쪽 엉덩이와 오른쪽 엉덩이 연결
    if len(keypoint_positions) > 12:
        if is_valid_keypoint(11) and is_valid_keypoint(12):
            cv2.line(frame, tuple(map(int, keypoint_positions[11])), tuple(map(int, keypoint_positions[12])), GREEN, 2)

    # 왼쪽 어깨 - 왼쪽 엉덩이 연결
    if len(keypoint_positions) > 11:
        if is_valid_keypoint(5) and is_valid_keypoint(11):
            cv2.line(frame, tuple(map(int, keypoint_positions[5])), tuple(map(int, keypoint_positions[11])), GREEN, 2)

    # 오른쪽 어깨 - 오른쪽 엉덩이 연결
    if len(keypoint_positions) > 12:
        if is_valid_keypoint(6) and is_valid_keypoint(12):
            cv2.line(frame, tuple(map(int, keypoint_positions[6])), tuple(map(int, keypoint_positions[12])), GREEN, 2)

    # 다리 연결 (왼쪽 엉덩이 - 왼쪽 무릎 - 왼쪽 발목)
    if len(keypoint_positions) > 15:
        if is_valid_keypoint(11) and is_valid_keypoint(13):
            cv2.line(frame, tuple(map(int, keypoint_positions[11])), tuple(map(int, keypoint_positions[13])), GREEN, 2)
        if is_valid_keypoint(13) and is_valid_keypoint(15):
            cv2.line(frame, tuple(map(int, keypoint_positions[13])), tuple(map(int, keypoint_positions[15])), GREEN, 2)

    # 다리 연결 (오른쪽 엉덩이 - 오른쪽 무릎 - 오른쪽 발목)
    if len(keypoint_positions) > 16:
        if is_valid_keypoint(12) and is_valid_keypoint(14):
            cv2.line(frame, tuple(map(int, keypoint_positions[12])), tuple(map(int, keypoint_positions[14])), GREEN, 2)
        if is_valid_keypoint(14) and is_valid_keypoint(16):
            cv2.line(frame, tuple(map(int, keypoint_positions[14])), tuple(map(int, keypoint_positions[16])), GREEN, 2)


mk_keypoints_video()
