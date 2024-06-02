# Core Packages
import streamlit as st
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os
import tempfile
import matplotlib.pyplot as plt
# import dlib
from tensorflow.keras.models import load_model
import Extract_images_from_video as extract

# Configuration de la page
st.set_page_config(
    page_title="FaceEmotion Insight",
    page_icon="😊"  # Utilisez un emoji Unicode pour l'icône de l'application
)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# face_detector = dlib.get_frontal_face_detector()

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

        image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

        if image_file is not None:
            our_image = Image.open(image_file)
            st.text("Original Image")
            st.image(our_image)

            enhance_type = st.sidebar.radio("Enhance Type", ["Original", "Gray-Scale", "Contrast", "Brightness", "Blurring"])
            if enhance_type == 'Gray-Scale':
                new_img = np.array(our_image.convert('RGB'))
                img = cv2.cvtColor(new_img, cv2.COLOR_RGB2GRAY)
                st.image(img, channels='GRAY')
            elif enhance_type == 'Contrast':
                c_rate = st.sidebar.slider("Contrast", 0.5, 3.5)
                enhancer = ImageEnhance.Contrast(our_image)
                img_output = enhancer.enhance(c_rate)
                st.image(img_output)
            elif enhance_type == 'Brightness':
                c_rate = st.sidebar.slider("Brightness", 0.5, 3.5)
                enhancer = ImageEnhance.Brightness(our_image)
                img_output = enhancer.enhance(c_rate)
                st.image(img_output)
            elif enhance_type == 'Blurring':
                new_img = np.array(our_image.convert('RGB'))
                blur_rate = st.sidebar.slider("Blurring", 0.5, 3.5)
                img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)
                blur_img = cv2.GaussianBlur(img, (11, 11), int(blur_rate))
                blur_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB)  # Convert back to RGB for display
                st.image(blur_img)

            task = ["Faces", "Emotions", "Cannize", "Cartonize"]
            feature_choice = st.sidebar.selectbox("Find Features", task)
            if st.button("Process"):
                if feature_choice == 'Faces':
                    result_img, result_faces = detect_faces(our_image)
                    st.image(result_img)
                    st.success(f"Found {len(result_faces)} faces")
                elif feature_choice == 'Emotions':
                    result_img = detect_emotions(our_image)
                    st.image(result_img)
                # elif feature_choice == 'Eyes':
                    # result_img = detect_eyes(our_image)
                    # st.image(result_img)
                elif feature_choice == 'Cartonize':
                    result_img = cartonize_image(our_image)
                    st.image(result_img)
                elif feature_choice == 'Cannize':
                    result_canny = cannize_image(our_image)
                    st.image(result_canny)

    elif choice == 'Emotion Detection':
        st.subheader("Emotion Detection")
        # Initialize session state if not already done

        if 'process_running' not in st.session_state:
            st.session_state.process_running = False
            st.session_state.images_extracted = False
        
        uploaded_file, video_name, video_name_with_extension, video_path = upload_video()

        if uploaded_file is not None and video_name_with_extension is not None:
            st.write(f"Selected video name: {video_name_with_extension}")
            output_extracted = f"Extracted_images/{video_name}"  # Output directory
            output_emotions = f"Emotions_detected/{video_name}"
            output_emotions_vd = "Emotions_videos"
            # output_emotions_vd_ld = f"Emotions_videos/{video_name_with_extension}"
            output_data_folder = 'Data_emotions'
            
            status_placeholder = st.empty()

            # Check if images have been extracted already
            if not st.session_state.images_extracted:
                status_placeholder.write(":hourglass_flowing_sand: Please wait while we process your video...")
                extract.extract_images(video_path, output_extracted)
                st.session_state.images_extracted = True
                status_placeholder.empty()

            if st.sidebar.button("Process !"):
                # Create a placeholder for the status message
                status_placeholder.write(":hourglass_flowing_sand: Please wait while we Detect Emotions of your Students ...")
                # Processing the video
                import Detect_emotions as emotions
                import detect_emotions_vd as emotions_vd
                emotions.process_images(output_extracted, output_emotions)
                # emotions_vd.analyze_video_emotions(video_path, output_emotions_vd, output_data_folder)
                
                status_placeholder.empty()

                # Show images of detected emotions in a 3x3 grid
                st.title("Detected Emotions")
                image_files = os.listdir(output_emotions)
                num_images = len(image_files)
                num_cols = 3
                num_rows = (num_images + num_cols - 1) // num_cols

                columns = st.columns(num_cols)
                for i in range(num_images):
                    columns[i % num_cols].image(f"Emotions_detected/{video_name}/{image_files[i]}", caption=image_files[i], use_column_width=True)

                # Show resulting video
                st.title("Resulting Video")
                video_file_path = os.path.join(output_emotions_vd, f'{video_name}.mp4')
                # emotion_video_path = output_emotions_vd_ld

                # video = os.listdir(output_emotions_vd_ld)

                if os.path.exists(video_file_path):
                    st.video(video_file_path)
                else:
                    st.error(f"Video file not found: {video_file_path}")

                # Read emotion data from text file and display pie chart
                emotion_data = {}
                emotion_file_path = f"{output_data_folder}/{video_name}.txt"
                if os.path.exists(emotion_file_path):
                    with open(emotion_file_path, 'r') as file:
                        for line in file:
                            emotion, percentage = line.strip().split(': ')
                            emotion_data[emotion] = float(percentage.strip('%'))

                    st.title("Emotion Distribution")
                    plot_emotion_pie_chart(emotion_data)

if __name__ == '__main__':
    main()
