"""
This script performs real-time facial emotion recognition using a classical
computer vision pipeline based on HOG (Histogram of Oriented Gradients) features
and a trained scikit-learn classifier.

Pipeline:
    Webcam frame → face detection → face crop → resize → grayscale
    → HOG feature extraction → ML classifier → emotion label overlay

HOG computes gradients of image brightness to capture the directions and
strengths of edges, which makes it a color-invariant descriptor of facial shape
and contour rather than raw pixel appearance.
"""

import cv2
import cvlib as cv
from skimage.feature import hog
import joblib
import numpy as np

# ===== Load trained model =====
# Make sure this path/name matches what you used when saving the model
model_filename = 'emotion_detection_model.pkl2'
loaded_model = joblib.load(model_filename)

# ===== Video capture =====
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# HOG params (must match training)
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = 'L2-Hys'

IMG_SIZE = 64  # must match training resize

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Detect faces in the frame
    faces, confidences = cv.detect_face(frame)

    for face, confidence in zip(faces, confidences):
        (start_x, start_y, end_x, end_y) = face

        # Safety: clip coords to frame boundaries
        h, w, _ = frame.shape
        start_x = max(0, start_x)
        start_y = max(0, start_y)
        end_x = min(w, end_x)
        end_y = min(h, end_y)

        # Crop the face
        face_crop = frame[start_y:end_y, start_x:end_x]

        # Skip if invalid crop
        if face_crop.size == 0:
            continue

        # Resize to same size as training
        face_resized = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))

        # Convert to grayscale
        gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)

        # HOG feature extraction (same params as training)
        features = hog(
            gray,
            orientations=HOG_ORIENTATIONS,
            pixels_per_cell=HOG_PIXELS_PER_CELL,
            cells_per_block=HOG_CELLS_PER_BLOCK,
            block_norm=HOG_BLOCK_NORM
        )

        # Reshape for model: (1, n_features)
        features = features.reshape(1, -1)

        # Predict emotion
        emotion = loaded_model.predict(features)[0]

        # Draw bounding box and label on the frame
        label = f'{emotion}'
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        cv2.putText(frame, label, (start_x, start_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
