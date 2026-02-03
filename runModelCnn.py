'''
performs real-time facial emotion recognition by detecting faces from a webcam feed, transforming each face into an LBP-based texture representation consistent with training, and classifying it using a fastai-trained convolutional neural network.
'''

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from fastai.vision.all import *
import cvlib as cv

# ----------------------------------------------------------
# LOAD YOUR TRAINED FASTAI MODEL (.pkl)
# ----------------------------------------------------------
learn = load_learner(r"C:\Users\sbval\PycharmProjects\MLSN-Team-6\emotion_classifier.pkl")


# ----------------------------------------------------------
# PREPROCESSING FUNCTION (MATCHES YOUR TRAINING PIPELINE)
# IMPORTANT: Your training images were inverted + LBP-based
# ----------------------------------------------------------
def preprocess_for_model(face_img):
    """
    Takes a BGR face image from webcam, applies:
    1. grayscale
    2. gaussian blur
    3. LBP transform
    4. INVERSION (because your training images look inverted)
    5. converts to fastai PILImage
    """

    # 1. Convert to grayscale
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

    # 2. Gaussian blur (your training code did this)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. LBP transform (must match your training params!)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")

    # 4. INVERT (your training samples were inverted)
    lbp = 255 - lbp

    # 5. Convert to PILImage for fastai
    img = PILImage.create(lbp.astype(np.uint8))

    return img


# ----------------------------------------------------------
# WEBCAM LOOP
# ----------------------------------------------------------
cap = cv2.VideoCapture(0)

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # detect faces via cvlib
    faces, confidences = cv.detect_face(frame)

    for face, conf in zip(faces, confidences):
        x1, y1, x2, y2 = face

        # crop face
        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            continue

        # -----------------------------
        # APPLY PREPROCESSING
        # -----------------------------
        img = preprocess_for_model(face_crop)

        # -----------------------------
        # PREDICT WITH FASTAI
        # -----------------------------
        pred_class, pred_idx, probs = learn.predict(img)
        label = str(pred_class)

        # draw bounding box + prediction
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2)

    # show output
    cv2.imshow("Emotion Detection (fastai)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
