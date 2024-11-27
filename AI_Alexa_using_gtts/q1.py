import os
import requests
import torch
import numpy as np
import re
import queue
import threading
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import speech_recognition as sr
from dotenv import load_dotenv
import whisper  # Import whisper package for speech recognition
import openai
from openai import OpenAI
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Load API key from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Ask ChatGPT function
def ask_chatgpt(prompt):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {OPENAI_API_KEY}',
    }

    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
        "temperature": 0.7,
    }

    try:
        response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        print("ChatGPT Response:", result)  # Debugging print
        return result['choices'][0]['message']['content'].strip()
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with OpenAI API: {e}")
        return f"Error communicating with OpenAI API: {e}"

# Function to record audio
def record_audio(audio_queue, energy, pause, dynamic_energy, verbose):
    r = sr.Recognizer()
    r.energy_threshold = energy
    r.pause_threshold = pause
    r.dynamic_energy_threshold = dynamic_energy

    with sr.Microphone(sample_rate=16000) as source:
        print("Listening...")
        while True:
            audio = r.listen(source)
            torch_audio = torch.from_numpy(np.frombuffer(audio.get_raw_data(), np.int16).flatten().astype(np.float32) / 32768.0)
            audio_queue.put_nowait(torch_audio)
            if verbose:
                print("Audio recorded.")

# Function to transcribe audio and detect wake word
def transcribe_forever(audio_queue, result_queue, audio_model, wake_word, verbose):
    while True:
        audio_data = audio_queue.get()
        result = audio_model.transcribe(audio_data, language='english')
        predicted_text = result["text"].strip()

        if verbose:
            print(f"Predicted Text: {predicted_text}")

        if predicted_text.lower().startswith(wake_word.lower()):
            pattern = re.compile(re.escape(wake_word), re.IGNORECASE)
            predicted_text = pattern.sub("", predicted_text).strip()
            if verbose:
                print(f"Detected wake word: {wake_word}. Processing: {predicted_text}")
            result_queue.put_nowait(predicted_text)
        else:
            if verbose:
                print("Wake word not detected, ignoring.")

# Function to generate a response using ChatGPT
def reply(result_queue, verbose):
    while True:
        question = result_queue.get()
        if verbose:
            print(f"Sending question to ChatGPT: {question}")

        prompt = f"Q: {question}?\nA:"
        answer = ask_chatgpt(prompt)

        if verbose:
            print(f"Received answer from ChatGPT: {answer}")

        # Convert the answer to speech
        try:
            mp3_obj = gTTS(text=answer, lang="en", slow=False)
            mp3_obj.save("reply.mp3")
            reply_audio = AudioSegment.from_mp3("reply.mp3")
            play(reply_audio)
        except Exception as e:
            print(f"Error in speech synthesis: {e}")

# Main function to start threads
def main(model="base", wake_word="hey computer", verbose=True):
    load_dotenv()

    # Load the Whisper model from whisper package
    audio_model = whisper.load_model(model)
    
    audio_queue = queue.Queue()
    result_queue = queue.Queue()

    # Start threads
    threading.Thread(target=record_audio, args=(audio_queue, 300, 0.8, False, verbose)).start()
    threading.Thread(target=transcribe_forever, args=(audio_queue, result_queue, audio_model, wake_word, verbose)).start()
    threading.Thread(target=reply, args=(result_queue, verbose)).start()

    while True:
        pass  # Keep the main thread alive

if __name__ == "__main__":
    main()
