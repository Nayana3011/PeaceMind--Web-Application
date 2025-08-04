PeaceMind – Real-Time Stress Detection Web Application
PeaceMind Web is an AI-powered web application that detects and classifies stress levels in users through facial emotion recognition using deep learning. It provides a seamless platform for users to upload images or use live video feeds, analyze facial expressions in real-time, and receive personalized stress management advice based on emotional patterns.
The system is built with a Flask backend and integrates a CNN-based model trained on facial emotion datasets. It supports secure login and registration via Firebase Authentication and enables emotion logging and session tracking for better stress monitoring over time.

Key Features

Facial Emotion Detection from uploaded images or webcam input

Real-time stress classification based on negative emotion frequency

Firebase Authentication for user login and registration

Session history tracking via backend APIs for long-term emotional monitoring

Stress relief suggestions based on detected emotional state

Tech Stack
Flask (Python) – Backend API and model hosting
TensorFlow / Keras – CNN model for emotion detection
OpenCV, dlib – Face detection and preprocessing
HTML/CSS/JavaScript – Frontend interface
Firebase Authentication – User auth system
Jinja2 Templates – Dynamic page rendering

REST APIs – For model predictions and session logging

Purpose
The web version of PeaceMind was designed to be accessible across platforms, allowing users to track and understand their emotional well-being without installing a mobile app. It supports research in mental health monitoring, especially for students, IT professionals, and those in high-stress environments.
