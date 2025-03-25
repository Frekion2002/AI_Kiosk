import cv2
import dlib

# dlib 얼굴 및 랜드마크 감지기 로드
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:\\Users\\aqtg6\\Downloads\\sp68_face_landmarks.dat\\sp68_face_landmarks.dat")


# 웹캠을 열고 영상 스트림 받기
cap = cv2.VideoCapture(0)

while True:
    # 프레임 읽기
    ret, img = cap.read()
    if not ret:
        break
    
    # 영상은 그레이스케일로 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 얼굴 감지
    faces = detector(gray)
    
    for face in faces:
        # 얼굴의 랜드마크 추출
        landmarks = predictor(gray, face)
        
        # 얼굴 영역 표시 (사각형)
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
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
    cv2.imshow('Real-time Facial Feature Detection', img)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 종료
cap.release()
cv2.destroyAllWindows()
