import cv2
from tensorflow.keras.models import load_model
import numpy as np
import os

def analyze_video_emotions(video_path, output_video_folder, output_data_folder):
    # Load emotion detection model
    emotion_model = load_model('fer2013_mini_XCEPTION.102-0.66.hdf5', compile=False)

    # Define emotions
    EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

    # Initialize face classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Function to detect emotions in an image
    def detect_emotions(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        emotions_count = {emotion: 0 for emotion in EMOTIONS}

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (64, 64))  # Resize image to (64, 64)
            roi_gray = roi_gray.astype("float") / 255.0
            roi_gray = np.expand_dims(roi_gray, axis=-1)  # Add dimension for color channel
            roi_gray = np.expand_dims(roi_gray, axis=0)  # Add dimension for batch

            preds = emotion_model.predict(roi_gray)[0]
            emotion_label = EMOTIONS[preds.argmax()]

            emotions_count[emotion_label] += 1

            # Draw a rectangle around the detected face
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Display the detected emotion
            cv2.putText(image, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        return image, emotions_count

    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Get video information (width, height, fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create folders to save the modified video and emotion data files
    if not os.path.exists(output_video_folder):
        os.makedirs(output_video_folder)
    if not os.path.exists(output_data_folder):
        os.makedirs(output_data_folder)

    # Get the filename of the original video
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Create a VideoWriter object to save the modified video in the output folder with the same filename
    output_path = os.path.join(output_video_folder, f'{video_name}.mp4')
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if out.isOpened():
        print(f"VideoWriter initialized successfully. Emotion video path: {output_path}")
    else:
        print("Failed to initialize VideoWriter.")

    # Initialize the dictionary to count emotions
    total_emotions_count = {emotion: 0 for emotion in EMOTIONS}
    total_faces_detected = 0

    # Read each frame of the video, detect emotions, and save the video with detected emotions
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if frame_count % 10 == 0:  # Process every second frame to speed up the process
                frame_with_emotions, emotions_count = detect_emotions(frame)
                out.write(frame_with_emotions)

                # Update the total count of emotions
                for emotion, count in emotions_count.items():
                    total_emotions_count[emotion] += count

                total_faces_detected += sum(emotions_count.values())

            frame_count += 1
        else:
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Calculate emotion percentages
    percentages_emotions = {emotion: (count / total_faces_detected) * 100 if total_faces_detected > 0 else 0 for emotion, count in total_emotions_count.items()}

    # Save results to a text file
    data_path = os.path.join(output_data_folder, f'{video_name}.txt')
    with open(data_path, 'w') as f:
        for emotion, percentage in percentages_emotions.items():
            f.write(f"{emotion}: {percentage:.2f}%\n")

    print("Emotions detected, video saved, and data file created successfully.")
