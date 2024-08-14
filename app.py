from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
import os
from dotenv import load_dotenv
from groq import Groq
import requests
import io
import base64

# Load environment variables
load_dotenv()

# Groq API key
groq_api_key = os.environ.get("GROQ_API_KEY", "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")
client = Groq(api_key=groq_api_key)

# ElevenLabs API key and voice ID
elevenlabs_api_key = os.environ.get("ELEVENLABS_API_KEY", "1b5618153ecf5a7ffef547170e95d29b")
elevenlabs_voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "yEWQhksxZDp1tYyEDLbm")

app = Flask(__name__)

LANGUAGES = {
    "Kannada": "kn",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Japanese": "ja"
}

recognizer = sr.Recognizer()

def translate_with_groq(text, target_language):
    system_prompt = f"""Translate the following text to {target_language}. 
    Only provide the translation, no additional notes or explanations."""

    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        temperature=0,
        max_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
    )
    return completion.choices[0].message.content

def generate_speech(text):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{elevenlabs_voice_id}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": elevenlabs_api_key
    }
    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        return base64.b64encode(response.content).decode('utf-8')
    else:
        raise Exception(f"ElevenLabs API error: {response.status_code} - {response.text}")

@app.route("/")
def index():
    return render_template("index.html", languages=LANGUAGES)

@app.route("/translate", methods=["POST"])
def translate():
    try:
        audio_file = request.files['audio']
        target_lang = request.form['language']

        # Recognize speech from the audio file
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
            english_text = recognizer.recognize_google(audio, language="en-US")

        # Translate the text
        translated = translate_with_groq(english_text, target_lang)

        # Generate speech from the translated text
        audio_output = generate_speech(translated)

        return jsonify({
            "original_text": english_text, 
            "translated_text": translated,
            "audio_output": audio_output
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)