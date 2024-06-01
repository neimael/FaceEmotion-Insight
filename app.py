# Core Packages
import streamlit as st
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os
import tempfile
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import Extract_images_from_video as extract
import Detect_emotions as emotions
import detect_emotions_vd as emotions_vd
from detect_emotions_vd import analyze_video_emotions

# Configuration de la page
st.set_page_config(
    page_title="FaceEmotion Insight",
    page_icon="😊"  # Utilisez un emoji Unicode pour l'icône de l'application
)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

try:
    emotion_model = load_model('fer2013_mini_XCEPTION.102-0.66.hdf5', compile=False)
except Exception as e:
    st.error(f"Error loading emotion detection model: {e}")

# Définir les émotions
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

def plot_emotion_pie_chart(emotion_data):
    labels = emotion_data.keys()
    sizes = emotion_data.values()
    
    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(sizes, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    # Customize legend
    ax.legend(wedges, labels,
              title="Emotions",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))
    
    st.pyplot(fig)

def detect_faces(our_image):
    try:
        new_img = np.array(our_image.convert('RGB'))
        img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Use the BGR image for conversion
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert back to RGB for display
        return img, faces 
    except Exception as e:
        st.error(f"Error detecting faces: {e}")
        return None, None

def detect_emotions(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Use the BGR image for conversion
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (64, 64))  # Redimensionner l'image à (64, 64)
        roi_gray = roi_gray.astype("float") / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=-1)  # Ajouter une dimension pour le canal de couleur
        roi_gray = np.expand_dims(roi_gray, axis=0)  # Ajouter une dimension pour le batch

        preds = emotion_model.predict(roi_gray)[0]
        emotion_label = EMOTIONS[preds.argmax()]
        # Dessiner un rectangle autour du visage détecté
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Afficher l'émotion détectée
        cv2.putText(img, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert back to RGB for display
    return img

def cartonize_image(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    cartoon = cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)  # Convert back to RGB for display
    return cartoon

def cannize_image(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
    img = cv2.GaussianBlur(img, (11, 11), 0)
    canny = cv2.Canny(img, 100, 150)
    return canny

def detect_faces_in_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return frame, faces

def upload_video():
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        video_name_with_extension = uploaded_file.name  # Save the name of the uploaded file
        video_name, extension = os.path.splitext(video_name_with_extension)  # Get the name without extension
        st.video(uploaded_file)
        
        # Save the uploaded video to the Videos directory
        video_dir = "Videos"
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        
        video_path = os.path.join(video_dir, video_name_with_extension)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return uploaded_file, video_name, video_name_with_extension, video_path
    else:
        return None, None, None, None

def main():
    """FaceEmotion Insight 😊"""
    st.title("FaceEmotion Insight 😊")

    activities = ["Face Detection", "Emotion Detection"]
    choice = st.sidebar.selectbox("Select Activity", activities)
    if choice == 'Face Detection':
        st.subheader("Face Detection")

        # Upload image
        image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

        if image_file is not None:
            our_image = Image.open(image_file)
            st.text("Original Image")
            st.image(our_image, use_column_width=True)

            enhance_type = st.sidebar.radio("Enhance Type", ["Original", "Gray-Scale", "Contrast", "Brightness", "Blurring"])
            if enhance_type == 'Gray-Scale':
                st.text("Gray-Scale Image")
                st.image(our_image.convert('L'), use_column_width=True)
            elif enhance_type == 'Contrast':
                c_rate = st.sidebar.slider("Contrast", 0.5, 3.5)
                enhancer = ImageEnhance.Contrast(our_image)
                img_output = enhancer.enhance(c_rate)
                st.image(img_output, use_column_width=True)
            elif enhance_type == 'Brightness':
                c_rate = st.sidebar.slider("Brightness", 0.5, 3.5)
                enhancer = ImageEnhance.Brightness(our_image)
                img_output = enhancer.enhance(c_rate)
                st.image(img_output, use_column_width=True)
            elif enhance_type == 'Blurring':
                blur_rate = st.sidebar.slider("Blurring", 0.5, 3.5)
                st.text("Blurred Image")
                st.image(our_image.filter(ImageFilter.GaussianBlur(blur_rate)), use_column_width=True)

            task = ["Faces", "Emotions", "Cartoonize"]
            feature_choice = st.sidebar.selectbox("Find Features", task)
            if st.button("Process"):
                if feature_choice == 'Faces':
                    st.text("Processing Faces...")
                    # Call function to detect faces
                elif feature_choice == 'Emotions':
                    st.text("Processing Emotions...")
                    # Call function to detect emotions
                elif feature_choice == 'Cartoonize':
                    st.text("Cartoonizing Image...")
                    # Call function to cartoonize image

    elif choice == 'Emotion Detection':
        st.subheader("Emotion Detection")

        # Upload video
        uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

        if uploaded_file is not None:
            video_name_with_extension = uploaded_file.name  # Save the name of the uploaded file
            video_name, extension = os.path.splitext(video_name_with_extension)  # Get the name without extension
            st.video(uploaded_file)

            output_video_folder = "Emotions_videos"
            output_data_folder = "Data_emotions"

            if st.sidebar.button("Process"):
                st.text("Processing Video Emotions...")
                video_path = os.path.join("Videos", video_name_with_extension)  # Save the uploaded video to the Videos directory
                output_path = analyze_video_emotions(video_path, output_video_folder, output_data_folder)
                st.success("Emotions Detected! Check the Resulting Video.")
                st.video(output_path)  # Display the resulting video


if __name__ == '__main__':
    main()
