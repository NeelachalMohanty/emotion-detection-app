import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from PIL import Image

# --- Load model and assets ---
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = load_model('Emotion_Detection.h5', compile=False)

class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emojis = ['😠', '🤢', '😨', '😄', '😢', '😲', '😐']

# --- Streamlit UI ---
st.set_page_config(page_title="Emotion Detection", layout="centered")

st.markdown(
    "<h1 style='text-align: center; color: #ff4b4b;'>😊 Real-Time Emotion Detection</h1>", 
    unsafe_allow_html=True
)

st.markdown(
    "<h3 style='text-align: center;'>Using Deep Learning + OpenCV + Streamlit</h3><hr>", 
    unsafe_allow_html=True
)

st.sidebar.title("🛠️ Choose Mode")
mode = st.sidebar.radio("Select Input Method:", ["📸 Webcam Snapshot", "🖼️ Upload Image"])


st.markdown("""
<style>
    .emotion-card {
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 12px;
        background-color: #f4f4f4;
        text-align: center;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- Emotion prediction logic ---
def predict_emotion(face_img):
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.resize(face_img, (64, 64))
    face_img = face_img.astype("float") / 255.0
    face_img = np.expand_dims(face_img, axis=-1)
    face_img = np.expand_dims(face_img, axis=0)
    preds = classifier.predict(face_img)[0]
    return preds

def show_prob_chart(preds):
    fig, ax = plt.subplots()
    bars = ax.bar(class_labels, preds, color=plt.cm.Paired(np.linspace(0, 1, len(class_labels))))
    ax.set_ylabel("Confidence")
    ax.set_ylim([0, 1])
    plt.xticks(rotation=45)
    for bar, score in zip(bars, preds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{score:.2f}', 
                ha='center', va='bottom', fontsize=9)
    st.pyplot(fig)

# --- Webcam Mode ---
if mode == "📸 Webcam Snapshot":
    st.markdown("### 📸 Capture Emotion via Webcam")
    picture = st.camera_input("Take a Snapshot Below 👇")

    if picture is not None:
        st.success("✅ Snapshot captured!")

        image_np = np.array(Image.open(picture).convert('RGB'))
        frame = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            st.warning("😕 No face detected.")
        else:
            for (x, y, w, h) in faces:
                roi = frame[y:y+h, x:x+w]
                preds = predict_emotion(roi)
                idx = np.argmax(preds)
                label = class_labels[idx]
                emoji = emojis[idx]
                confidence = preds[idx]

                # Draw rectangle and overlay emoji + label + confidence directly on face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                overlay_text = f"{label} ({confidence*100:.1f}%)"
                cv2.putText(frame, overlay_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                st.markdown(f"<h2 style='text-align:center'>{emoji} {label} ({confidence*100:.1f}%)</h2>", unsafe_allow_html=True)


            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="🧠 Emotion Prediction", channels="RGB")

            st.markdown("### 📊 Emotion Probabilities")
            show_prob_chart(preds)

# --- Image Upload Mode ---
elif mode == "🖼️ Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image_np = np.array(Image.open(uploaded_file).convert('RGB'))
        frame = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            st.warning("😕 No face detected.")
        else:
            for (x, y, w, h) in faces:
                roi = frame[y:y+h, x:x+w]
                preds = predict_emotion(roi)
                idx = np.argmax(preds)
                label = class_labels[idx]
                emoji = emojis[idx]
                confidence = preds[idx]

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                overlay_text = f"{emoji} {label} ({confidence*100:.1f}%)"
                cv2.putText(frame, overlay_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="🧠 Emotion Prediction", channels="RGB")
            st.markdown("### 🔍 Full Emotion Probabilities")
            show_prob_chart(preds)


