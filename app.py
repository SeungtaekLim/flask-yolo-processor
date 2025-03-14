from flask import Flask, request, jsonify, send_file
import os
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import subprocess

app = Flask(__name__)

# 업로드 폴더 설정
UPLOAD_FOLDER = 'uploads/'
RESULT_FOLDER = 'results/'  # 결과 폴더 설정
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
fourcc_h264 = cv2.VideoWriter_fourcc(*'X264')  # H.264 코덱

# YOLO 모델 로드 (여기서는 'yolo11m-pose.pt' 모델을 로드)
model = YOLO("model/yolo11m-pose.pt")

# 업로드된 파일을 저장하고 바로 처리하는 엔드포인트
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify(message="파일이 없습니다."), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify(message="파일 이름이 비어 있습니다."), 400
    
    # 파일 확장자 체크 (이미지 또는 비디오)
    file_extension = file.filename.split('.')[-1].lower()

    # 파일 저장 경로
    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    
    try:
        # 폴더가 없으면 생성
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        
        # 파일 저장
        file.save(filename)
        print(f"파일 저장 성공: {filename}")  # 로그 추가

        # 이미지 또는 비디오 파일 처리
        if file_extension in ['mp4', 'avi', 'mov']:
            processed_video_path = process_video(filename)  # 비디오 처리 함수 호출
            return send_file(processed_video_path, as_attachment=True)  # 처리된 비디오 반환
        else:
            processed_image_path = process_image(filename)  # 이미지 처리 함수 호출
            return send_file(processed_image_path, as_attachment=True)  # 처리된 이미지 반환

    except Exception as e:
        print(f"오류 발생: {str(e)}")  # 오류 로그
        return jsonify(message=f"파일 저장 중 오류 발생: {str(e)}"), 500


def process_video(video_path):
    try:
        # metadata = get_video_rotation(video_path)
        # print(f"rotation Metadata: {metadata}")  # 메타데이터 출력

        video_path = process_video_with_rotation(video_path)

        # YOLO로 비디오 파일 전체 처리 (비디오 결과를 그대로 반환)
        results = model(video_path, stream=True)  # generator 형태로 결과를 반환
        
        # 결과를 저장할 임시 비디오 파일 경로 (이름에 "_after" 추가)
        result_video_name = video_path.split('/')[-1].split('.')[0] + '_after.mp4'
        result_video_path = os.path.join(app.config['RESULT_FOLDER'], result_video_name)
        
        # 비디오 캡처 객체로 입력 비디오 열기
        cap = cv2.VideoCapture(video_path)
        
        # 비디오의 프레임 크기 (너비, 높이) 가져오기
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out = cv2.VideoWriter(result_video_path, fourcc_h264, 20.0, (frame_width, frame_height))
        
        # generator를 순차적으로 처리하여 각 프레임을 처리
        for result in results:
            # 각 프레임에 대한 YOLO 결과 이미지 얻기
            annotated_frame = result.plot()  # YOLO 결과 이미지 (프레임에 주석 달린 이미지)
            # 이미지 크기 조정 (원본 비디오 크기와 일치하도록)
            # annotated_frame_resized = cv2.resize(annotated_frame, (frame_width, frame_height))
            # 주석이 달린 이미지를 비디오로 저장
            out.write(annotated_frame)
        
        # 비디오 저장 완료
        out.release()
        cap.release()  # 비디오 캡처 객체 해제

        print(f"결과 동영상 저장 성공: {result_video_path}")

        return result_video_path  # 처리된 비디오 경로 반환

    except Exception as e:
        print(f"비디오 처리 중 오류 발생: {str(e)}")


def process_image(image_path):
    try:
        # YOLO 모델로 이미지 처리
        results = model(image_path)

        # 결과 이미지 저장 경로 (RESULT_FOLDER에 저장)
        result_image_name = image_path.split('/')[-1].split('.')[0] + '_after.' + image_path.split('.')[-1]
        result_image_path = os.path.join(app.config['RESULT_FOLDER'], result_image_name)

        # YOLO 결과로 주석이 달린 이미지를 생성 (numpy.ndarray 반환)
        annotated_image = results[0].plot()  # 결과 이미지를 얻음
        
        # numpy.ndarray에서 BGR -> RGB로 변환
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
        # numpy.ndarray -> PIL.Image 변환
        pil_image = Image.fromarray(annotated_image_rgb)
        
        # 이미지를 저장
        pil_image.save(result_image_path)
        print(f"결과 이미지 저장 성공: {result_image_path}")  # 로그 추가

        return result_image_path  # 처리된 이미지 경로 반환

    except Exception as e:
        print(f"이미지 처리 중 오류 발생: {str(e)}")

# def get_image_metadata(image_path):
#     try:
#         # ffmpeg 명령어로 이미지 메타데이터 추출 (표준 출력)
#         command = ['ffmpeg', '-i', image_path]
#         result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

#         # stderr에 메타데이터가 포함되어 있음
#         if result.stderr:
#             return result.stderr  # stderr에 출력된 메타데이터를 반환
        
#         # 만약 stdout에 결과가 나오면 그것도 반환
#         return result.stdout
#     except Exception as e:
#         print(f"메타데이터 추출 중 오류 발생: {str(e)}")
#         return None
    
def get_video_rotation(image_path):
    try:
        # ffmpeg 명령어로 메타데이터를 추출
        command = ['ffmpeg', '-i', image_path]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # stderr에서 회전 정보 추출 (rotation 값을 찾아서 반환)
        if result.stderr:
            for line in result.stderr.splitlines():
                if "displaymatrix: rotation" in line:
                    # 회전 정보만 추출
                    rotation = line.split('rotation of')[-1].strip()
                    return float(rotation.split()[0])  # 회전 값 반환 (-90.00 등)

        # 회전 정보가 없으면 None 반환
        return None

    except Exception as e:
        print(f"메타데이터 추출 중 오류 발생: {str(e)}")
        return None

def rotate_video(image_path, rotation):
    # 영상 파일 열기
    cap = cv2.VideoCapture(image_path)
    
    # 비디오 파일이 정상적으로 열렸는지 확인
    if not cap.isOpened():
        print("비디오 파일을 열 수 없습니다.")
        return None  # None 반환

    # 비디오의 가로, 세로 크기 및 프레임 레이트 정보 가져오기
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 출력 파일 이름 생성 (입력 파일명에 "_rotated" 추가)
    output_path = os.path.splitext(image_path)[0] + f"_rotated.mp4"
    
    # VideoWriter 객체 설정
    fourcc_h264 = cv2.VideoWriter_fourcc(*'X264')
    out = cv2.VideoWriter(output_path, fourcc_h264, 20.0, (height, width))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 회전 각도에 따라 회전 처리
        if rotation == -90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  # 시계방향 90도
        elif rotation == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 시계방향 90도
        
        # 회전한 프레임을 출력 비디오 파일에 작성
        out.write(frame)
    
    # 비디오 객체 해제
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    return output_path  # 회전된 비디오 경로 반환

def process_video_with_rotation(image_path):
    rotation = get_video_rotation(image_path)
    if rotation is None:
        print("회전 정보가 없습니다. 원본 동영상을 그대로 반환합니다.")
        return image_path  # 회전 정보가 없으면 원본 동영상 그대로 반환
    else:
        print(f"회전 정보: {rotation} degrees")
        rotated_video_path = rotate_video(image_path, rotation)
        return rotated_video_path  # 회전된 동영상 경로 반환

if __name__ == '__main__':
    # 폴더가 없으면 생성
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    if not os.path.exists(RESULT_FOLDER):
        os.makedirs(RESULT_FOLDER)
    
    # Flask 앱 실행
    app.run(debug=True, port=5000)
