# Smart Learning Assistant

A gesture and expression recognition system that adapts to your learning state and responds accordingly. This intelligent learning companion uses computer vision and speech recognition to create a more interactive learning experience.

## Features (Version 1.0)

- **Facial Expression Recognition**: Recognizes 7 different facial expressions (Happy, Sad, Angry, Disgust, Fear, Neutral, Surprise)
- **Hand Gesture Recognition**: Detects hand gestures including Thumbs Up and Thumbs Down
- **Voice Command Recognition**: Uses OpenAI's Whisper model to understand voice commands
- **Automated Responses**: Provides appropriate feedback based on detected emotional and gestural cues
- **Application Integration**: Can open applications (like Spotify) based on voice commands
- **Streamlit Web Interface**: Easy-to-use graphical interface to interact with the system

## Demo

When the system detects:
- **Happy face + Thumbs Up**: Encourages the user to continue studying
- **Sad face + Thumbs Down**: Suggests taking a break and listens for voice commands
  - Saying "Open Spotify" will automatically launch the Spotify application

## Installation

1. Install required packages:
```
pip install -r requirements.txt
```

2. Download the required models:
   - Place the "Facial Expression CNN.pth" model in the root directory
   - Place the "Gesture Recognition Task Guide.task" file in the root directory

## Usage

Run the Streamlit application:
```
streamlit run app.py
```

Follow the instructions in the application:
1. Click "Load Models" to initialize the system
2. Click "Start" to begin the webcam feed
3. Interact with the system using facial expressions and hand gestures
4. Use voice commands when prompted

## Requirements

- Python 3.8+
- Webcam
- Microphone
- GPU recommended for better performance

## Dependencies

- OpenCV
- PyTorch
- Mediapipe
- Whisper
- Streamlit
- pyttsx3
- SoundDevice
- SoundFile
- SoundCard
- NumPy

## Coming Soon: Version 2.0!

We're excited to announce that Version 2.0 is currently in development and will include these advanced features:

- **Real-time Chat Interface**: Have natural conversations with the learning assistant
- **Interactive Q&A**: Ask questions about your study material and receive instant answers
- **Advanced Learning Analytics**: Track your focus, emotions, and learning patterns over time
- **Personalized Learning Suggestions**: Receive content and break recommendations based on your learning state
- **Multi-modal Integration**: Enhanced integration with various learning platforms and applications
- **Improved UI/UX**: More intuitive and responsive user interface

Stay tuned for updates! Version 2.0 will be released soon.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
