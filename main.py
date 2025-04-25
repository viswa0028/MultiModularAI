import tempfile
import cv2
import numpy as np
import soundfile as sf
import sounddevice as sd
import soundcard as sc
import whisper
import mediapipe as mp
from mediapipe.tasks.python.vision import GestureRecognizer as GR
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
model = whisper.load_model("base")
base_options = python.BaseOptions(model_asset_path='./Gesture Recognition Task Guide.task')
options = vision.GestureRecognizerOptions(base_options=base_options,running_mode=vision.RunningMode.VIDEO)
recognizer = vision.GestureRecognizer.create_from_options(options)

def record_audio(seconds=5, samplerate=16000, file_name=None):
    if file_name is None:
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        file_name = temp_file.name
        temp_file.close()

    print(f"Recording for {seconds} seconds...")
    recording = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    print("Finished recording")
    # Save as WAV file
    sf.write(file_name, recording, samplerate)

    return transcript_audio(file_name)
def transcript_audio(file_name):
    text = model.transcribe(file_name)
    print(text['text'])
# record_audio()

webcam = cv2.VideoCapture(0)
frame_counter = 0
while True:
    ret,frame = webcam.read()
    # cv2.imshow("something",frame)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    gesture_recognition_result = recognizer.recognize_for_video(mp_image,frame_counter)
    frame_counter= frame_counter+1
    cv2.imshow('Frame',frame)
    if gesture_recognition_result.gestures:
        for gestures in gesture_recognition_result:
            gesture = gestures[0].category_name
            cv2.putText(frame,gestures,(10,30),fontFace=cv2.FONT_ITALIC,fontScale=2)
            cv2.imshow("Gesture",frame)
    if cv2.waitKey(1)& 0xFF == ord('q'):
        webcam.release()
        break
webcam.release()
cv2.destroyAllWindows()