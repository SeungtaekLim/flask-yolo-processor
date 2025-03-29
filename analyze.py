KEYPOINT_NAMES = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear", 
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow", 
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip", 
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

def analyze(all_keypoints_data):
    for frame_idx, keypoints_data in enumerate(all_keypoints_data):
        print(f"\n Frame [{frame_idx + 1}]:")
        print_keypoints(keypoints_data)

def print_keypoints(keypoints_data):
    # keypoints 출력 (x, y 좌표와 이름)
    for i in range(17):  # 17개 키포인트 전부를 출력
        x, y = keypoints_data[i]  # 현재 키포인트의 좌표
        
        # 키포인트가 (0, 0)인 경우 "None"을 출력하고 그렇지 않으면 (x, y) 값을 출력
        if x == 0 and y == 0:
            coordinates_text = "None"
        else:
            coordinates_text = f"({int(x)}, {int(y)})"
        
        # 키포인트 이름과 좌표를 출력
        print(f"{KEYPOINT_NAMES[i]}: {coordinates_text}")
