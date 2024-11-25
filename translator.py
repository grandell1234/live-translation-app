import re
from openai import OpenAI
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from pydub import AudioSegment
from pydub.playback import play
import io
import threading
import queue
import time

# Replace with your OpenAI API Key.
OPENAI_API_KEY = "sk-1234"

client = OpenAI(api_key=OPENAI_API_KEY)

# Modify depending on what you want your mic to be.
device_index = 2

sample_rate = 44100

audio_queue = queue.Queue()

english_transcriptions = []
spanish_translations = []

def record_audio():
    try:
        while True:
            print("[DEBUG] Starting audio recording...")
            audio_data = sd.rec(int(5 * sample_rate), samplerate=sample_rate, channels=1, dtype='int16', device=device_index)
            sd.wait()
            print("[DEBUG] Audio recording complete.")

            wav_file = io.BytesIO()
            wav.write(wav_file, sample_rate, audio_data)
            wav_file.seek(0)

            audio_segment = AudioSegment.from_wav(wav_file)
            mean_amplitude = np.mean(np.abs(audio_data))
            print(f"[DEBUG] Mean amplitude: {mean_amplitude}")

            if mean_amplitude > 15:
                audio_queue.put(audio_segment)
                print("[DEBUG] Audio segment added to the queue.")
            else:
                print("[WARNING] No significant audio detected. Skipping this segment.")
    except Exception as e:
        print(f"[ERROR] Exception in record_audio: {e}")

def process_audio():
    try:
        while True:
            if not audio_queue.empty():
                print("[DEBUG] Retrieving audio segment from the queue.")
                audio_segment = audio_queue.get()
                mp3_file_path = "output.mp3"

                try:
                    print("[DEBUG] Exporting audio segment to MP3 format...")
                    audio_segment.export(mp3_file_path, format="mp3")
                except Exception as e:
                    print(f"[ERROR] Failed to export audio: {e}")
                    continue

                try:
                    with open(mp3_file_path, "rb") as audio_file:
                        print("[DEBUG] Sending audio for transcription...")
                        transcription = client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file,
                            prompt="."
                        )
                except Exception as e:
                    print(f"[ERROR] Exception during transcription: {e}")
                    continue

                transcription_text = transcription.text.strip()
                print(f"[DEBUG] Received transcription: {transcription_text}")

                if not transcription_text or re.match(r'^\s*\.?\s*(\.\s*|\s*\.)*\s*$', transcription_text):
                    print("[WARNING] No valid transcription available. Skipping.")
                    continue

                english_transcriptions.append(transcription_text)

                try:
                    print("[DEBUG] Sending transcription for translation...")
                    completion = client.chat.completions.create(
                        model="gpt-4o",
                        temperature=0,
                        messages=[
                            {"role": "system", "content": "You are a translator which translates English to Spanish. Every piece of text you receive after this you need to translate into Spanish. Only respond with the translated sentence:"},
                            {"role": "user", "content": transcription_text}
                        ]
                    )
                except Exception as e:
                    print(f"[ERROR] Exception during translation: {e}")
                    continue

                translation = completion.choices[0].message.content
                print(f"[DEBUG] Translation received: {translation}")
                spanish_translations.append(translation)

                try:
                    speech_file_path = "./speech.mp3"
                    print("[DEBUG] Converting translation to speech...")
                    response = client.audio.speech.create(
                        model="tts-1",
                        voice="alloy",
                        input=translation
                    )
                    response.stream_to_file(speech_file_path)
                except Exception as e:
                    print(f"[ERROR] Exception during text-to-speech conversion: {e}")
                    continue

                try:
                    print("[DEBUG] Playing translated speech audio...")
                    audio = AudioSegment.from_mp3(speech_file_path)
                    play(audio)
                except Exception as e:
                    print(f"[ERROR] Exception while playing audio: {e}")

                with open("english_transcriptions.txt", "w") as eng_file:
                    eng_file.write("\n".join(english_transcriptions))
                with open("spanish_translations.txt", "w") as span_file:
                    span_file.write("\n".join(spanish_translations))
    except Exception as e:
        print(f"[ERROR] Exception in process_audio: {e}")

print("[DEBUG] Starting threads...")
recording_thread = threading.Thread(target=record_audio)
processing_thread = threading.Thread(target=process_audio)

recording_thread.start()
processing_thread.start()

recording_thread.join()
processing_thread.join()