import cv2
import numpy as np
import dlib
from tensorflow import keras
import time
import platform
import winsound
import os
from datetime import datetime

# صوت التنبيه على ويندوز
def beep():
    winsound.PlaySound("beep.wav", winsound.SND_FILENAME | winsound.SND_ASYNC)

# تحميل النموذج
model = keras.models.load_model("model.keras")

# مكونات الكشف
predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# إنشاء مسارات حفظ الصور
os.makedirs("dataset/open", exist_ok=True)
os.makedirs("dataset/closed", exist_ok=True)

# دالة لقص العين
def crop_eye(image, eye_points):
    x_coords = [pt[0] for pt in eye_points]
    y_coords = [pt[1] for pt in eye_points]
    x1, x2 = min(x_coords) - 20, max(x_coords) + 20
    y1, y2 = min(y_coords) - 20, max(y_coords) + 20
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, image.shape[1])
    y2 = min(y2, image.shape[0])
    return image[y1:y2, x1:x2], (x1, y1, x2, y2)

# دالة معالجة العين قبل التنبؤ
def preprocess_eye(eye_img):
    eye = cv2.resize(eye_img, (24, 24))
    eye = eye.astype("float32")
    if len(eye.shape) == 2:
        eye = cv2.merge([eye] * 3)
    return eye

# متغيرات العد
eye_closed_start_time = None
alert_triggered = False
alert_duration_seconds = 5


cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 0)

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        right_eye_points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)]
        left_eye_points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)]

        try:
            left_eye_img, (lx1, ly1, lx2, ly2) = crop_eye(frame, left_eye_points)
            right_eye_img, (rx1, ry1, rx2, ry2) = crop_eye(frame, right_eye_points)

            left_eye = preprocess_eye(left_eye_img)
            
            left_input = np.expand_dims(left_eye, axis=0)/255.
            right_eye = preprocess_eye(right_eye_img)
            right_input = np.expand_dims(right_eye, axis=0)/255.

            left_pred = model.predict(left_input)[0][0]
            right_pred = model.predict(right_input)[0][0]

            left_label = "Open" if left_pred > 0.5 else "Closed"
            right_label = "Open" if right_pred > 0.5 else "Closed"

            left_color = (255, 255, 255) if left_label == "Open" else (0, 0, 255)
            right_color = (255, 255, 255) if right_label == "Open" else (0, 0, 255)

            cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (255, 255, 255), 2)
            cv2.putText(frame, f"L: {left_label}", (lx1, ly1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, left_color, 2)

            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 255, 255), 2)
            cv2.putText(frame, f"R: {right_label}", (rx1, ry1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, right_color, 2)

            # حفظ الصور
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # لإنتاج اسم متميز
            if left_label == "Open":
                cv2.imwrite(f"dataset/open/{timestamp}_left.png", left_eye_img)
            else:
                cv2.imwrite(f"dataset/closed/{timestamp}_left.png", left_eye_img)

            if right_label == "Open":
                cv2.imwrite(f"dataset/open/{timestamp}_right.png", right_eye_img)
            else:
                cv2.imwrite(f"dataset/closed/{timestamp}_right.png", right_eye_img )

            # تنبيه عند بقاء العينين مغمضتين
            if left_label == "Closed" and right_label == "Closed":
                if eye_closed_start_time is None:
                    eye_closed_start_time = time.time()
                    alert_triggered = False
                else:
                    elapsed = time.time() - eye_closed_start_time
                    cv2.putText(frame, f"Eyes closed for {int(elapsed)}s",
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    if elapsed >= alert_duration_seconds  :
                        alert_triggered = True
                        print("⚠️ تنبيه: العينان مغلقتان لأكثر من 5 ثواني!")
                        if platform.system() == "Windows":
                            beep()
                        else:
                            print('\a')
            else:
                eye_closed_start_time = None
                alert_triggered = False

        except Exception as e:
            print("خطأ أثناء معالجة العين:", e)

    cv2.imshow("Eye State Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
