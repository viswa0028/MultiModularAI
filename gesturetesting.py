import numpy as np
import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='./Gesture Recognition Task Guide.task')
options = vision.GestureRecognizerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO)
recognizer = vision.GestureRecognizer.create_from_options(options)

webcam = cv2.VideoCapture(0)
frame_counter = 0

while True:
    ret, frame = webcam.read()
    if not ret:
        break
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    gesture_recognition_result = recognizer.recognize_for_video(mp_image, frame_counter)
    frame_counter += 1
    cv2.imshow('Frame', frame)
    if gesture_recognition_result.gestures:
        gesture_name = gesture_recognition_result.gestures[0][0].category_name
        confidence = round(gesture_recognition_result.gestures[0][0].score, 2)
        text = f"{gesture_name} ({confidence})"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()