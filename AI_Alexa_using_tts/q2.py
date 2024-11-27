from pydub import AudioSegment
from pydub.playback import play
import speech_recognition as sr
import whisper
import queue
import os
import threading
import torch
import numpy as np
import re
import requests
import json
from dotenv import load_dotenv, find_dotenv
import openai
import click

_ = load_dotenv(find_dotenv())  # Load .env file
openai.api_key = os.environ['OPENAI_API_KEY']

@click.command()
@click.option("--model", default="base", help="Model to use", type=click.Choice(["tiny", "base", "small", "medium", "large"]))
@click.option("--english", default=False, help="Whether to use the English model", is_flag=True, type=bool)
@click.option("--energy", default=300, help="Energy level for the mic to detect", type=int)
@click.option("--pause", default=0.8, help="Pause time before entry ends", type=float)
@click.option("--dynamic_energy", default=False, is_flag=True, help="Flag to enable dynamic energy", type=bool)
@click.option("--wake_word", default="hey computer", help="Wake word to listen for", type=str)
@click.option("--verbose", default=False, help="Whether to print verbose output", is_flag=True, type=bool)
def main(model, english, energy, pause, dynamic_energy, wake_word, verbose):
    if model != "large" and english:
        model = model + ".en"
    audio_model = whisper.load_model(model)
    audio_queue = queue.Queue()
    result_queue = queue.Queue()

    threading.Thread(target=record_audio, args=(audio_queue, energy, pause, dynamic_energy)).start()
    threading.Thread(target=transcribe_forever, args=(audio_queue, result_queue, audio_model, english, wake_word, verbose)).start()
    threading.Thread(target=reply_with_tts, args=(result_queue, verbose)).start()

    while True:
        print(result_queue.get())

def record_audio(audio_queue, energy, pause, dynamic_energy):
    r = sr.Recognizer()
    r.energy_threshold = energy
    r.pause_threshold = pause
    r.dynamic_energy_threshold = dynamic_energy

    with sr.Microphone(sample_rate=16000) as source:
        print("Listening...")
        while True:
            audio = r.listen(source)
            torch_audio = torch.from_numpy(np.frombuffer(audio.get_raw_data(), np.int16).flatten().astype(np.float32) / 32768.0)
            audio_data = torch_audio
            audio_queue.put_nowait(audio_data)

def transcribe_forever(audio_queue, result_queue, audio_model, english, wake_word, verbose):
    while True:
        audio_data = audio_queue.get()
        if english:
            result = audio_model.transcribe(audio_data, language='english')
        else:
            result = audio_model.transcribe(audio_data)

        predicted_text = result["text"]

        if predicted_text.strip().lower().startswith(wake_word.strip().lower()):
            pattern = re.compile(re.escape(wake_word), re.IGNORECASE)
            predicted_text = pattern.sub("", predicted_text).strip()
            punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
            predicted_text = predicted_text.translate({ord(i): None for i in punc})
            if verbose:
                print("You said the wake word.. Processing {}...".format(predicted_text))

            result_queue.put_nowait(predicted_text)
        else:
            if verbose:
                print("You did not say the wake word.. Ignoring")

def reply_with_tts(result_queue, verbose):
    while True:
        question = result_queue.get()
        prompt = f"Q: {question}?\nA:"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {openai.api_key}',
        }
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
            "temperature": 0.5,
        }

        try:
            response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            answer = result['choices'][0]['message']['content'].strip()

            # Convert the text answer into speech using an alternative TTS service (e.g., ElevenLabs or Azure)
            tts_response = requests.post(
                "https://api.elevenlabs.io/v1/speech/generate",  # Replace with TTS API endpoint
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {os.getenv('TTS_API_KEY')}"
                },
                json={"text": answer, "voice": "Rachel"}  # Replace "Rachel" with your preferred voice
            )
            tts_response.raise_for_status()
            audio_data = tts_response.content

            # Play the TTS response
            with open("reply.wav", "wb") as audio_file:
                audio_file.write(audio_data)
            reply_audio = AudioSegment.from_file("reply.wav", format="wav")
            play(reply_audio)

        except requests.exceptions.RequestException as e:
            if verbose:
                print(f"TTS API Error: {e}")
            reply_fallback(verbose)

def reply_fallback(verbose):
    fallback_audio = AudioSegment.silent(duration=1000) + AudioSegment.from_text("An error occurred. Please try again.", lang="en")
    play(fallback_audio)

main()
