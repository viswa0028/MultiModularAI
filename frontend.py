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
import streamlit as st
import threading
import time
import os
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase


# CNN Model definition
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


# Initialize session state variables
if 'started' not in st.session_state:
    st.session_state.started = False
if 'expression' not in st.session_state:
    st.session_state.expression = "None"
if 'gesture' not in st.session_state:
    st.session_state.gesture = "None"
if 'latest_action' not in st.session_state:
    st.session_state.latest_action = "None"
if 'stop_signal' not in st.session_state:
    st.session_state.stop_signal = False


def load_models():
    # Load ML models
    st.session_state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.session_state.ExpressionModel = Convolution(num_classes=7)
    st.session_state.ExpressionModel.load_state_dict(torch.load('./Facial Expression CNN.pth'))
    st.session_state.ExpressionModel.to(st.session_state.device)
    st.session_state.ExpressionModel.eval()

    # Initialize TTS engine
    st.session_state.ttsx_engine = pyttsx3.init()

    # Load Whisper model
    st.session_state.whisper_model = whisper.load_model("base")

    # Load Gesture Recognizer
    base_options = python.BaseOptions(model_asset_path='./Gesture Recognition Task Guide.task')
    options = vision.GestureRecognizerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO)
    st.session_state.recognizer = vision.GestureRecognizer.create_from_options(options)

    st.session_state.models_loaded = True


def record_audio(seconds=5, samplerate=16000, file_name=None):
    if file_name is None:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        file_name = temp_file.name
        temp_file.close()

    status_placeholder = st.empty()
    status_placeholder.info(f"Recording for {seconds} seconds...")
    recording = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    status_placeholder.info("Finished recording. Processing audio...")
    sf.write(file_name, recording, samplerate)

    return transcript_audio(file_name)


def transcript_audio(file_name):
    if 'whisper_model' in st.session_state:
        text = st.session_state.whisper_model.transcribe(file_name)
        return text['text']
    return "Whisper model not loaded"


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
        img_tensor = transform(pil_image).unsqueeze(0).to(st.session_state.device)
        with torch.no_grad():
            output = st.session_state.ExpressionModel(img_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
        class_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
        return class_labels[predicted_class]
    except Exception as e:
        st.error(f"Error in processing expression: {e}")
        return "Unknown"


def speak_text(text):
    st.session_state.ttsx_engine.say(text)
    st.session_state.ttsx_engine.runAndWait()


def open_spotify():
    try:
        # For Mac
        if os.name == 'posix' and os.uname().sysname == 'Darwin':
            subprocess.run(["open", "/Applications/Spotify.app"])
        # For Windows
        elif os.name == 'nt':
            subprocess.run(["start", "spotify"], shell=True)
        # For Linux
        elif os.name == 'posix':
            subprocess.run(["spotify"])

        st.session_state.latest_action = "Opened Spotify"
    except Exception as e:
        st.error(f"Failed to open Spotify: {e}")
        st.session_state.latest_action = f"Error opening Spotify: {e}"


def process_video_stream():
    webcam = cv2.VideoCapture(0)
    frame_counter = 0

    while not st.session_state.stop_signal:
        ret, frame = webcam.read()
        if not ret:
            break

        mp_image = mp.Image(data=frame, image_format=mp.ImageFormat.SRGB)
        gesture_result = st.session_state.recognizer.recognize_for_video(mp_image, frame_counter)
        frame_counter += 1

        # Process expression
        expression = process_expression(frame)
        st.session_state.expression = expression

        # Process gesture
        gesture = "None"
        if gesture_result.gestures:
            gesture_name = gesture_result.gestures[0][0].category_name
            gesture = gesture_name
            st.session_state.gesture = gesture

            # Handle expression and gesture combinations
            if expression == 'Happy' and gesture == 'Thumb_Up':
                st.session_state.latest_action = "Happy + Thumb Up detected"
                threading.Thread(target=speak_text, args=("I guess you are happy lets focus on studying",)).start()
                time.sleep(2)  # Give time for speech to complete

            if expression == 'Sad' and gesture == 'Thumb_Down':
                st.session_state.latest_action = "Sad + Thumb Down detected"
                threading.Thread(target=speak_text, args=(
                'I guess you are not happy with the way things are proceeding, lets take a Break!',)).start()
                time.sleep(2)  # Give time for speech to complete

                # Record audio and check for Spotify command
                st.session_state.latest_action = "Listening for command..."
                recording = record_audio()
                st.session_state.latest_action = f"Heard: {recording}"

                if "open spotify" in recording.lower():
                    open_spotify()

        # Display the frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if 'frame_placeholder' in st.session_state:
            st.session_state.frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

    webcam.release()
    st.session_state.started = False


# Streamlit UI
st.title("Gesture and Expression Recognition")

# Sidebar for controls
st.sidebar.header("Controls")

# Load models button
if 'models_loaded' not in st.session_state or not st.session_state.models_loaded:
    if st.sidebar.button("Load Models"):
        with st.spinner("Loading models..."):
            load_models()
        st.success("Models loaded successfully!")

# Start/Stop buttons
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Start", disabled=not (
            'models_loaded' in st.session_state and st.session_state.models_loaded) or st.session_state.started):
        st.session_state.started = True
        st.session_state.stop_signal = False
        st.session_state.frame_placeholder = st.empty()
        threading.Thread(target=process_video_stream).start()

with col2:
    if st.button("Stop", disabled=not st.session_state.started):
        st.session_state.stop_signal = True
        st.warning("Stopping the system...")

# Manual trigger for audio recording
if st.sidebar.button("Record Audio (5 sec)",
                     disabled=not ('models_loaded' in st.session_state and st.session_state.models_loaded)):
    transcript = record_audio()
    st.sidebar.write(f"Transcript: {transcript}")

    if "open spotify" in transcript.lower():
        open_spotify()

# Display status
st.sidebar.header("Status")
st.sidebar.write(f"Expression: {st.session_state.expression}")
st.sidebar.write(f"Gesture: {st.session_state.gesture}")
st.sidebar.write(f"Latest Action: {st.session_state.latest_action}")

# Main display area
if not st.session_state.started:
    st.info("Click 'Start' in the sidebar to begin the video stream")

# Instructions
st.markdown("""
## Instructions:
1. Click on 'Load Models' in the sidebar to initialize all required models
2. Click 'Start' to begin the webcam feed
3. The system will detect your facial expressions and hand gestures
4. Try these combinations:
   - Happy face + Thumbs Up → System will encourage studying
   - Sad face + Thumbs Down → System will suggest a break and listen for commands
   - Say "Open Spotify" when prompted to open Spotify
5. Click 'Stop' to end the webcam feed
""")

# Display system information
st.sidebar.header("System Info")
st.sidebar.write(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")