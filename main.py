import cv2
import numpy as np
import dlib
from tensorflow import keras

# تحميل النموذج المدرب
model = keras.models.load_model("model.keras")

# تحميل كاشف الوجه والمعالم
predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# دالة لقص العين
def crop_eye(image, eye_points):
    x_coords = [pt[0] for pt in eye_points]
    y_coords = [pt[1] for pt in eye_points]
    x1, x2 = min(x_coords) - 25, max(x_coords) + 25
    y1, y2 = min(y_coords) - 25, max(y_coords) + 25
    return image[y1:y2, x1:x2], (x1, y1, x2, y2)

# دالة تجهيز الصورة للنموذج
def preprocess_eye(eye_img):
    eye = cv2.resize(eye_img, (24, 24))  # غيّر الحجم حسب ما يناسب نموذجك
    eye = eye.astype("float32") / 255.0
    if len(eye.shape) == 2:
        eye = cv2.merge([eye]*3)
    return np.expand_dims(eye, axis=0)

# تشغيل الكاميرا
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # نقاط العينين
        left_eye_points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)]
        right_eye_points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)]

        try:
            # قص العين اليسرى واليمنى + مواقعها
            left_eye_img, (lx1, ly1, lx2, ly2) = crop_eye(frame, left_eye_points)
            right_eye_img, (rx1, ry1, rx2, ry2) = crop_eye(frame, right_eye_points)

            # تجهيز العينين للنموذج
            left_input = preprocess_eye(left_eye_img)
            right_input = preprocess_eye(right_eye_img)

            # التنبؤ بالحالة
            left_pred = model.predict(left_input)[0][0]
            right_pred = model.predict(right_input)[0][0]

            # تحويل التنبؤ إلى نص
            left_label = "Open" if left_pred > 0.5 else "Closed"
            right_label = "Open" if right_pred > 0.5 else "Closed"


            left_color = (255, 255, 255) if left_label == "Open" else (0, 0, 255)
            right_color = (255, 255, 255) if right_label == "Open" else (0, 0, 255)

            # رسم مستطيل حول العين اليسرى
        
            cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (255, 255, 255), 2)
            cv2.putText(frame, f"L: {left_label}", (lx1, ly1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, left_color, 2)

            # رسم مستطيل حول العين اليمنى
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 255, 255), 2)
            cv2.putText(frame, f"R: {right_label}", (rx1, ry1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, right_color, 2)
        except Exception as e:
            print("خطأ أثناء تحليل العين:", e)

    # عرض الفيديو
    cv2.imshow("Eye State Detection", frame)

    # الخروج بـ "q"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# إنهاء
cap.release()
cv2.destroyAllWindows()
