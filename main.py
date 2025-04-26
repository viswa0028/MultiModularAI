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
import torch
from torchvision import transforms
import torch.nn
from PIL import Image
import pyttsx3
import subprocess
class Convolution(torch.nn.Module):
    def __init__(self, num_classes):
        super(Convolution, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.poolinglayer = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = torch.nn.Linear(18432, 512)
        self.fc2 = torch.nn.Linear(512, num_classes)
        self.dropout = torch.nn.Dropout(0.5)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.poolinglayer(self.relu(self.conv2(x)))
        x = self.poolinglayer(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ExpressionModel = Convolution(num_classes=7)
ExpressionModel.load_state_dict(torch.load('./Facial Expression CNN.pth'))
ExpressionModel.to(device)
ExpressionModel.eval()
ttsx_engine =pyttsx3.init()
model = whisper.load_model("base")
base_options = python.BaseOptions(model_asset_path='./Gesture Recognition Task Guide.task')
options = vision.GestureRecognizerOptions(base_options=base_options,running_mode=vision.RunningMode.VIDEO)
recognizer = vision.GestureRecognizer.create_from_options(options)

def record_audio(seconds=5, samplerate=16000, file_name=None):
    if file_name is None:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        file_name = temp_file.name
        temp_file.close()

    print(f"Recording for {seconds} seconds...")
    recording = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    print("Finished recording")
    sf.write(file_name, recording, samplerate)

    return transcript_audio(file_name)
def transcript_audio(file_name):
    text = model.transcribe(file_name)
    return text['text']
# record_audio()

def process_expression(frame):
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        img_tensor = transform(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = ExpressionModel(img_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
        class_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
        return class_labels[predicted_class]
    except Exception as e:
        print(f"Error in processing expression: {e}")
        return "Unknown"
webcam = cv2.VideoCapture(0)
frame_counter = 0
while True:
    ret,frame = webcam.read()
    if not ret:
        break
    mp_image = mp.Image(data=frame,image_format = mp.ImageFormat.SRGB)
    gesture_result = recognizer.recognize_for_video(mp_image,frame_counter)
    frame_counter = frame_counter+1
    expression = process_expression(frame)
    gesture = None
    cv2.putText(frame,expression,(60,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
    if gesture_result.gestures:
        gesture_name = gesture_result.gestures[0][0].category_name
        gesture = gesture_name
        text = f'{gesture_name}'
        cv2.putText(frame,text,(150,130),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
        if expression == 'Happy' and gesture =='Thumb_Up':
            ttsx_engine.say("I guess you are happy lets focus on studying")
            ttsx_engine.runAndWait()
            break
        if expression == 'Sad' and gesture == 'Thumb_Down':  #
            ttsx_engine.say('I guess you are not happy with the way things are proceeding, lets take a Break!')
            # ttsx_engine.runAndWait()
            recoding = record_audio()
            if "open spotify" in recoding.lower():
                subprocess.call(
                    ['/usr/bin/open', '-n', '-a', '/Applications/Spotify.app'])
            break
    cv2.imshow('frame',frame)

    if cv2.waitKey(1)& 0xFF == ord('q'):
        webcam.release()
        break
webcam.release()
cv2.destroyAllWindows()