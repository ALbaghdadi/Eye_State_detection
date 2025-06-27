# Eye_State_detection
AI model to detect  eyes state weather Open/Close


# Eye State Detection (Eye Open/Closed Monitor)
![image](https://github.com/user-attachments/assets/9f02ee59-ba66-428f-aced-c48fc6e37e30)



This project detects whether a person's eyes are open or closed in real-time using a webcam. If both eyes remain closed for more than 5 seconds, an alert is triggered (with sound on Windows).

## 🚀 Features
- Real-time eye detection using `dlib` facial landmarks.
- Classifies eye state using a pre-trained neural network model.
- Automatically saves images of eyes (open/closed) with timestamped filenames.
- Triggers an alert if the user’s eyes remain closed for a specified duration.

## 📁 Project Structure
```
eye-state-detection/
│
├── main2.py                     # Main script
├── model.keras                  # Trained Keras model for eye state detection
├── shape_predictor_68_face_landmarks.dat  # dlib's facial landmarks model
├── dataset/
│   ├── open/                    # Saved images of open eyes
│   └── closed/                  # Saved images of closed eyes
├── beep.wav                     # Alert sound file (Windows only)
├── requirements.txt             # List of dependencies
└── README.md                    # Project documentation
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/eye-state-detection.git
   cd eye-state-detection
   ```

2. **Create virtual environment** *(optional but recommended)*:
   ```bash
   python -m venv tf_env
   source tf_env/Scripts/activate  # On Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download dlib model**:
   Place `shape_predictor_68_face_landmarks.dat` in the project folder. Download from:  
   [http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

5. **Make sure you have the pre-trained Keras model** `model.keras` in the root directory.

## ▶️ Usage

Run the main detection script:
```bash
python main2.py
```

- Press **`q`** to quit the application.
- Images are saved to `dataset/open` and `dataset/closed`.

## ✅ Requirements

See `requirements.txt`, but core libraries include:

- OpenCV
- NumPy
- TensorFlow / Keras
- dlib

## 🧠 Model Training (Optional)

To train your own model to detect eye states:

1. Collect a dataset of labeled eye images (open and closed) — as the app already does automatically.
2. Use the following Python code to train a simple neural network:

```python
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical

def load_data(folder_open, folder_closed):
    X, y = [], []
    for folder, label in [(folder_open, 1), (folder_closed, 0)]:
        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            img = cv2.imread(path)
            if img is not None:
                img = cv2.resize(img, (24, 24))
                X.append(img)
                y.append(label)
    return np.array(X), np.array(y)

X, y = load_data("dataset/open", "dataset/closed")
X = X.astype("float32") / 255.0
y = to_categorical(y, 2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(24, 24, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

model.save("model.keras")
```

## 🔊 Sound Alert
- Sound alert works **only on Windows** using `winsound`.
- Customize the alert by replacing `beep.wav`.

## 📷 Camera Note
- `cv2.VideoCapture(1)` is used — you may need to change to `0` if the default webcam doesn't work.

## 📄 License
This project is for educational and research purposes.

