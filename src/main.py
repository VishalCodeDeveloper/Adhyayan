from flask import Flask, request, jsonify, send_from_directory, render_template
from audio_processing import extract_audio, get_audio_duration, transcribe_audio
from text_analysis import analyze_text
from video_analysis import analyze_video
import datetime
import os

app = Flask(__name__)

# Ensure the uploads directory exists
os.makedirs('uploads', exist_ok=True)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    video_file = request.files['file']
    if video_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save video file temporarily
    video_path = f"uploads/{video_file.filename}"
    video_file.save(video_path)

    # Process audio
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_file = f"output_audio_{timestamp}.wav"
    extract_audio(video_path, audio_file)
    
    # Get audio duration and transcribe
    audio_duration = get_audio_duration(audio_file)
    transcribed_text = transcribe_audio(audio_file)

    # Analyze text and video
    print(f"\n\n")
    print(f"transcribed_text : ", transcribed_text)
    print(f"\n\n")
    text_analysis = analyze_text(transcribed_text, audio_duration)

    # print("text_analysis.sentiment : ", text_analysis.sentiment)

    for keyword in text_analysis.items():
            print(keyword)
            print("\n\n\n")

    video_analysis = analyze_video(video_path)

    print(f"Emotion changes: {video_analysis}")

    for time, emotion in video_analysis.emotion_changes.items():
         print(f"Time: {time} - Emotion: {emotion}")

    for time, detected in video_analysis.eye_contact.items():
        print(f"Time: {time} - Eye Contact: {'Yes' if detected else 'No'}")
         
    
    for second, gestures in video_analysis.hand_gestures.items():
        print(f"Second: {second} - Gestures")
        for gesture, count in gestures.items():
            print(f"{gesture}: {count}")

    results(transcribed_text, text_analysis, video_analysis)
    # return jsonify({
    #     "transcribed_text": transcribed_text,
    #     "text_analysis": text_analysis,
    #     "video_analysis": video_analysis
    # })

def results(transcribed_text, text_analysis, video_analysis):
    # Simulated analysis data    
    analysis_results = {
        "transcribed_text": transcribed_text,
        "text_analysis": text_analysis,
        "video_analysis": video_analysis
    }
    return render_template('results.html', results=analysis_results)

if __name__ == '__main__':
    app.run(debug=True)
