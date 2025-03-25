import cv2
import dlib
import numpy as np

def detect_facial_features(image_path):
    # dlib 얼굴 및 랜드마크 감지기 로드
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("C:\\Users\\aqtg6\\Downloads\\sp68_face_landmarks.dat\\sp68_face_landmarks.dat")
    
    # OpenCV 얼굴 감지기 로드
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # 이미지 로드
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # OpenCV 얼굴 감지
    faces_cv = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces_cv:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # dlib 얼굴 감지
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        
        # 이목구비 감지 및 표시
        for i in range(36, 48):  # 눈, 코, 입의 랜드마크 포인트
            x, y = landmarks.part(i).x, landmarks.part(i).y
            cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
        
        # 코 표시
        for i in range(27, 36):
            x, y = landmarks.part(i).x, landmarks.part(i).y
            cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
        
        # 입 표시
        for i in range(48, 68):
            x, y = landmarks.part(i).x, landmarks.part(i).y
            cv2.circle(img, (x, y), 2, (255, 255, 0), -1)
    
    # 결과 출력
    cv2.imshow('Facial Feature Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 테스트 실행
detect_facial_features('D:\\ComputerVision\\Kiosk_face_detection\\Karina.jpg')
