import requests
import base64
import os

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("API_KEY")

url = "https://api.inworld.ai/tts/v1/voice"

headers = {
    "Authorization": API_KEY,
    "Content-Type": "application/json"
}

payload = {
  "text": 'text',
  "voiceId": "Clive",
  "modelId": "inworld-tts-1.5-max",
  "timestampType": "WORD",
  "speakingRate": 1,
  "temperature": 1
}

def speak(text):
    payload["text"] = text
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()

    result = response.json()
    audio_content = base64.b64decode(result['audioContent'])

    with open("output.mp3", "wb") as f:
        f.write(audio_content)

    os.system("afplay output.mp3")
    

