import cv2
from tensorflow.keras.models import model_from_json
import numpy as np
from collections import Counter

# Load model architecture and weights
with open("static/model/emotiondetector.json", "r") as json_file:
    model = model_from_json(json_file.read())
model.load_weights("static/model/emotiondetector.h5")

# Load Haar Cascade for face detection
haar_file = cv2.data.haarcascades + 'static/model/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Map labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

def preprocess_image(image):
    """Preprocess the image for model prediction."""
    feature = np.array(image).reshape(1, 48, 48, 1)
    return feature / 255.0

# Stress-related emotions and threshold
stress_emotions = {'angry', 'disgust', 'fear', 'sad'}
threshold = 10  # Number of stress-indicating emotions before flagging stress
emotion_counts = Counter()  # Keeps track of detected emotions

# Start webcam
webcam = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = webcam.read()
        if not ret:
            break
        
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            # Extract face region
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = preprocess_image(face)
            
            # Predict emotion
            pred = model.predict(face)
            emotion = labels[np.argmax(pred)]
            
            # Count the emotion
            emotion_counts[emotion] += 1
            
            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Check if stress emotions exceed threshold
        stress_count = sum(emotion_counts[emotion] for emotion in stress_emotions)
        if stress_count >= threshold:
            stress_status = "Stressed"
        else:
            stress_status = "Not Stressed"
        
        # Display stress status
        cv2.putText(frame, f"Stress Status: {stress_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Display the frame
        cv2.imshow("Stress Detector", frame)
        
        # Exit on pressing 'ESC'
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    # Release resources
    webcam.release()
    cv2.destroyAllWindows()
