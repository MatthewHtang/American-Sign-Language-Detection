# American Sign Language Detection And Speech Detection

## Demo
![ASL Demo](sampleImage/sampledemo.gif)

📌Overview
This project is a real-time American Sign Language (ASL) Detection System that uses Computer Vision and Machine Learning to recognize ASL hand gestures and convert them into both text and speech.

The system captures live webcam input, detects hand gestures, classifies ASL letters using a trained model, constructs sentences over time, and converts the final sentence into realistic speech using the Inworld AI Text-to-Speech API.

💾 Programming Language
Python 3

👾 Computer Vision & Machine Learning
    OpenCV (cv2)
    cvzone
    NumPy
    Teachable Machine (TensorFlow/Keras model)
💬 Text-to-Speech
    Inworld AI TTS API
    Requests
    Base64

🛠️ Utility Libraries
    math
    time
    os
    python-dotenv

⚙️ Installation
1️⃣ Clone the repository

>>>> git clone https://github.com/yourusername/American-Sign-Language-Detection.git
>>>> cd American-Sign-Language-Detection

2️⃣ Create virtual environment
>>>> python -m venv .venv

Activate environment: 
macOS/Linux : source .venv/bin/activate
Windows : .venv\Scripts\activate

3️⃣ Install dependencies
pip install -r requirements.txt

🔑 API Setup
INWORLD_API_KEY=Basic YOUR_API_KEY_HERE


User Manual
| Key | Function                   |
| --- | -------------------------- |
| `q` | Quit program               |
| `c` | Clear sentence             |



