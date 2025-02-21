from flask import Flask, request, jsonify
import os
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

app = Flask(__name__)

# 업로드 폴더 설정
UPLOAD_FOLDER = 'uploads/'
RESULT_FOLDER = 'results/'  # 결과 폴더 설정
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# YOLO 모델 로드 (여기서는 자세 추정 모델을 로드)
model = YOLO("yolo11m-pose.pt")

# 업로드된 파일을 저장하고 바로 처리하는 엔드포인트
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify(message="파일이 없습니다."), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify(message="파일 이름이 비어 있습니다."), 400
    
    # 파일 저장 경로
    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    
    try:
        # 폴더가 없으면 생성
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        
        # 파일 저장
        file.save(filename)
        print(f"파일 저장 성공: {filename}")  # 로그 추가
        
        # 이미지 파일 처리 (YOLO 모델을 사용하여 결과 생성)
        results = model(filename)
        
        # 결과를 저장할 폴더가 없으면 생성
        result_path = os.path.join(app.config['RESULT_FOLDER'], 'image_results')
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        
        # YOLO 결과로 주석이 달린 이미지를 생성 (numpy.ndarray 반환)
        annotated_image = results[0].plot()  # 결과 이미지를 얻음
        
        # numpy.ndarray에서 BGR -> RGB로 변환
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
        # numpy.ndarray -> PIL.Image 변환
        pil_image = Image.fromarray(annotated_image_rgb)
        
        # 결과 이미지 저장 경로
        result_image_path = os.path.join(result_path, file.filename)
        
        # 이미지를 저장
        pil_image.save(result_image_path)
        print(f"결과 이미지 저장 성공: {result_image_path}")  # 로그 추가

        return jsonify(
            message="이미지 처리 완료. 결과 이미지를 확인하세요.",
            result_image=result_image_path
        ), 200

    except Exception as e:
        print(f"오류 발생: {str(e)}")  # 오류 로그
        return jsonify(message=f"파일 저장 중 오류 발생: {str(e)}"), 500

if __name__ == '__main__':
    # 폴더가 없으면 생성
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    if not os.path.exists(RESULT_FOLDER):
        os.makedirs(RESULT_FOLDER)
    
    # Flask 앱 실행
    app.run(debug=True, port=5000)
