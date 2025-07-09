import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Load model and face detector
classifier = load_model('Emotion_Detection.h5', compile=False)
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Labels and emojis (optional print in console)
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emojis = ['😠', '🤢', '😨', '😄', '😢', '😲', '😐']

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = classifier.predict(roi)[0]
        idx = np.argmax(preds)
        label = class_labels[idx]
        confidence = preds[idx]
        emoji = emojis[idx]

        # Show label and confidence on frame
        text = f"{label} ({confidence*100:.1f}%)"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        # Print in console too
        print(f"Detected: {label} {emoji} ({confidence*100:.1f}%)")

    cv2.imshow("🎥 Live Emotion Detection - Press Q to Quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
