# runModel.py  (cvlib version, with guards)
import cv2
import cvlib as cv
import joblib
import numpy as np

MODEL = "emotion_detection_model.pkl"
loaded_model = joblib.load(MODEL)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    result = cv.detect_face(frame)
    if not result:
        # nothing detected this frame; just show it
        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    faces, confidences = result

    for face, conf in zip(faces, confidences):
        start_x, start_y, end_x, end_y = face

        # clip to frame bounds
        h, w = frame.shape[:2]
        start_x = max(0, min(start_x, w - 1))
        end_x   = max(0, min(end_x,   w - 1))
        start_y = max(0, min(start_y, h - 1))
        end_y   = max(0, min(end_y,   h - 1))
        if end_x <= start_x or end_y <= start_y:
            continue

        face_crop = frame[start_y:end_y, start_x:end_x]
        if face_crop.size == 0:
            continue

        face_resize = cv2.resize(face_crop, (100, 100))
        face_flat = face_resize.flatten().reshape(1, -1)

        emotion = loaded_model.predict(face_flat)[0]

        label = f"{emotion}"
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        cv2.putText(frame, label, (start_x, max(0, start_y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()