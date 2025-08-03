from flask import Flask, request, jsonify, render_template
import cv2
import logging
from datetime import datetime
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
import firebase_admin
from firebase_admin import credentials, db

from firebase_connection import read_user_by_credentials,create_user

from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Step 1: Load the model structure (architecture)
with open('static/model/emotiondetector.json', 'r') as json_file:
    model_json = json_file.read()

# Step 2: Create model from the loaded architecture
model = model_from_json(model_json)

# Step 3: Load the model weights
model.load_weights('static/model/emotiondetector.h5')

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('static/model/haarcascade_frontalface_default.xml')

# Emotion labels
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

# Global variable to track consecutive negative emotions
consecutive_negative_count = 0  # Add this at the top of your script
NEGATIVE_EMOTIONS = {'sad', 'angry', 'fear', 'disgust'}  # Define negative emotions

# @app.route('/predict', methods=['POST'])
# def predict():
#     """
#     Route to process the uploaded frame and return detected emotions and stress status.
#     """
#     global consecutive_negative_count

#     # Retrieve the uploaded frame
#     file = request.files.get('frame')
#     if not file:
#         return jsonify({'error': 'No frame uploaded'}), 400

#     # Decode the uploaded frame
#     npimg = np.frombuffer(file.read(), np.uint8)
#     frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect faces in the frame
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
#     detected_emotions = []
#     face_data = []  # Store face coordinates and emotions
#     negative_detected = False

#     for (x, y, w, h) in faces:
#         # Extract the face ROI
#         detected_face = gray[y:y + h, x:x + w]
#         detected_face = cv2.resize(detected_face, (48, 48))

#         # Preprocess the face for the model
#         img_pixels = image.img_to_array(detected_face) / 255.0
#         img_pixels = np.expand_dims(img_pixels, axis=0)

#         # Predict emotion
#         predictions = model.predict(img_pixels)
#         emotion = emotions[np.argmax(predictions[0])]
#         detected_emotions.append(emotion)
        
#         # Draw rectangle and label
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
#         cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#         # Check for negative emotions
#         if emotion in NEGATIVE_EMOTIONS:
#             negative_detected = True

#         # Add face data (coordinates and emotion)
#         face_data.append({
#             'x': int(x),
#             'y': int(y),
#             'width': int(w),
#             'height': int(h),
#             'emotion': emotion
#         })

#     # Update consecutive negative emotion count
#     if negative_detected:
#         consecutive_negative_count += 1
#     else:
#         consecutive_negative_count = 0

#     print(f"Detected emotions: {detected_emotions}")  # Debug detected emotions
#     # Determine stress status
#     if consecutive_negative_count >= 5:
#         return jsonify({'status': 'stress_detected', 'faces': face_data}), 200
 
#     return jsonify({'status': 'ok', 'faces': face_data}), 200

@app.route('/predict', methods=['POST'])
def predict():
    global consecutive_negative_count  # Use the global counter

    file = request.files.get('frame')
    if not file:
        return jsonify({'error': 'No frame uploaded'}), 400

    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    print(f"Number of faces detected: {len(faces)}")

    detected_emotions = []
    negative_detected = False

    for (x, y, w, h) in faces:
        detected_face = gray[y:y + h, x:x + w]
        detected_face = cv2.resize(detected_face, (48, 48))

        # Preprocess the face image
        img_pixels = image.img_to_array(detected_face) / 255.0
        img_pixels = np.expand_dims(img_pixels, axis=0)

        # Predict emotion
        predictions = model.predict(img_pixels)
        print(f"Predictions: {predictions}")  # Debug the predictions
        emotion = emotions[np.argmax(predictions[0])]
        detected_emotions.append(emotion)

        if emotion in NEGATIVE_EMOTIONS:
            negative_detected = True

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        #cv2.putText(frame, f"Face {faces.index((x, y, w, h)) + 1}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    # Update consecutive negative count
    if negative_detected:
        consecutive_negative_count += 1
    else:
        consecutive_negative_count = 0

    print(f"Detected emotions: {detected_emotions}")  # Debug detected emotions
    print(f"Consecutive negative count: {consecutive_negative_count}")  # Debug negative count

    # Check if stress condition is met
    if consecutive_negative_count >= 5:
        return jsonify({'status': 'stress_detected', 'emotions': detected_emotions}), 200

    response = {
        'status': 'ok',
        'emotions': detected_emotions,
    }
    return jsonify(response)



@app.route('/',methods=['GET','POST'])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        user = read_user_by_credentials(email, password)
        if user:
            return render_template('user_dashboard.html', user=user)
        else:
             return "<script>alert('Invalid Credentials!'); window.location = '/';</script>"

    return render_template('login.html')

# cred = credentials.Certificate('serviceAccountKey.json')  # Replace with your correct path
# try:
#     firebase_admin.initialize_app(cred, {
#         'databaseURL': 'https://stressdetection-3c1a2-default-rtdb.firebaseio.com'
#     })
#     print("Firebase initialized successfully.")
# except Exception as e:
#     print(f"Error: {e}")

# ref = db.reference('registration')

# # Login API Endpoint
# @app.route('/login', methods=['POST'])
# def api_login():
#     data = request.get_json()
#     email = data.get('email')
#     password = data.get('password')

#     users = ref.get()
#     if users:
#         if isinstance(users, dict):
#             for user_id, user_data in users.items():
#                 if user_data.get('email') == email and user_data.get('password') == password:
#                     return jsonify({
#                         "status": "success",
#                         "user": {
#                             "id": user_id,
#                             "name": user_data.get('name'),
#                             "email": user_data.get('email'),
#                             "phno": user_data.get('phno')
#                         }
#                     }), 200
#     return jsonify({"status": "failure", "message": "Invalid email or password"}), 401

@app.route('/login', methods=['POST'])
def api_login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    # Dummy check
    if email == "test@gmail.com" and password == "123456":
        return jsonify({
            "status": "success",
            "user": {
                "id": "1",
                "name": "Test User",
                "email": email
            }
        })
    elif email == "newuser@gmail.com" and password == "asdfgh":
        return jsonify({
            "status": "success",
            "user": {
                "id": "1",
                "name": "Test User",
                "email": email
            }
        })
    else:
        return jsonify({"status": "failure"}), 401

@app.route('/register',methods=['GET','POST'])
def register():
    if request.method == "POST":
        firstname = request.form['first_name']
        lastname = request.form['last_name']
        phone = request.form['phone']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        if password != confirm_password:
            return "<script>alert('Password and Confirm Password do not match!'); window.location = '/register';</script>"

        
        create_user(firstname, lastname, phone, email, password)
        return "<script>alert('Registeration Sucess'); window.location = '/';</script>"


    return render_template('register.html')

@app.route('/register', methods=['POST'])
def api_register():
    data = request.get_json()
    
    firstname = data.get('first_name')
    lastname = data.get('last_name')
    phone = data.get('phone')
    email = data.get('email')
    password = data.get('password')
    confirm_password = data.get('confirm_password')

    if not all([firstname, lastname, phone, email, password, confirm_password]):
        return jsonify({"status": "failure", "message": "All fields are required"}), 400

    if password != confirm_password:
        return jsonify({"status": "failure", "message": "Passwords do not match"}), 400

    try:
        create_user(firstname, lastname, phone, email, password)
        return jsonify({"status": "success", "message": "Registration successful"}), 200
    except Exception as e:
        return jsonify({"status": "failure", "message": f"Error: {str(e)}"}), 500



@app.route('/ping', methods=['GET'])
def ping():
    return "Pong!", 200

@app.route('/user_dashboard')
def user_dashboard():
    return render_template('user_dashboard.html')

@app.route('/stress_detection')
def stress_detection():
    return render_template('camera.html')

# Enable logging to console
logging.basicConfig(level=logging.INFO)

@app.route('/trigger-action', methods=['POST'])
def trigger_action():
    try:
        data = request.get_json()

        # Validate incoming data
        user_id = data.get("userId")
        action = data.get("action")

        if not user_id or not action:
            return jsonify({"status": "error", "message": "Missing 'userId' or 'action'"}), 400

        # Simulate action (e.g., logging, DB update, etc.)
        timestamp = datetime.now().isoformat()
        logging.info(f"[{timestamp}] Action received from {user_id}: {action}")

        # Return success response
        return jsonify({
            "status": "success",
            "message": f"Action '{action}' triggered for user {user_id}",
            "timestamp": timestamp
        }), 200

    except Exception as e:
        logging.exception("Error while handling trigger")
        return jsonify({"status": "error", "message": str(e)}), 500
    
@app.route('/check_status')
def check_status():
    return jsonify({"status": "server running"}), 200

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)



# if __name__ == '__main__':
#     app.run(debug=True)
