import subprocess
import wave
import json
import vosk

def extract_audio(video_path, audio_path):
    command = ['ffmpeg', '-i', video_path, '-ac', '1', '-ar', '16000', audio_path]
    subprocess.run(command)

def get_audio_duration(audio_path):
    command = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
               '-of', 'default=noprint_wrappers=1:nokey=1', audio_path]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return float(result.stdout.strip())

def transcribe_audio(audio_path):
    try:
        model = vosk.Model("C:\\Work\\Sarvarth\\Development\\AI-SpeechRecognition\\VoskModel\\VoskModelENIN") 
        recognizer = vosk.KaldiRecognizer(model, 16000)

        # Open the audio file
        with wave.open(audio_path, "rb") as wf:
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
                print("Audio file must be in WAV format mono PCM 16kHz")
                return ""

            # Read audio chunks and process audio chunk data
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if recognizer.AcceptWaveform(data):
                    continue

        # Get the final result
        result = recognizer.FinalResult()
        return json.loads(result)['text']
    except Exception as e:
        print(f"Error during transcript : {e}")
        return ""
