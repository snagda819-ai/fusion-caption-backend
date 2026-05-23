from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import whisper
import subprocess
import os
import re
import threading
import time

app = Flask(__name__)
CORS(app)

# LOAD WHISPER MODEL
model = whisper.load_model("base")

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ---------------- FORMAT SRT TIME ----------------
def format_srt_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)

    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


# ---------------- CLEAN TEXT ----------------
def clean_text(text):
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = re.sub(r"[{}]", "", text)
    return text.strip()


# ---------------- AUTO CLEANUP ----------------
def cleanup(files):

    time.sleep(15)

    for f in files:

        if os.path.exists(f):

            try:
                os.remove(f)

            except:
                pass


# ---------------- HOME ----------------
@app.route("/")
def home():
    return "Fusion Caption AI Running"


# ---------------- MAIN API ----------------
@app.route("/upload", methods=["POST"])
def upload_video():

    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    video = request.files["video"]

    if video.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # 🔥 LANGUAGE OPTION
    language = request.form.get("language", "en")

    input_video = os.path.join(UPLOAD_FOLDER, "input_video.mp4")
    fixed_video = os.path.join(OUTPUT_FOLDER, "fixed_video.mp4")
    subtitle_file = os.path.join(OUTPUT_FOLDER, "captions.srt")
    final_video = os.path.join(OUTPUT_FOLDER, "final_video.mp4")

    # CLEAN OLD FILES
    for path in [input_video, fixed_video, subtitle_file, final_video]:
        if os.path.exists(path):
            try:
                os.remove(path)
            except:
                pass

    # SAVE INPUT
    video.save(input_video)

    # ---------------- STEP 1: NORMALIZE VIDEO ----------------
    normalize_cmd = [
        "ffmpeg",
        "-y",
        "-i", input_video,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-c:a", "aac",
        "-pix_fmt", "yuv420p",
        fixed_video
    ]

    normalize = subprocess.run(
        normalize_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if normalize.returncode != 0:
        return jsonify({
            "error": "Video normalization failed",
            "details": normalize.stderr
        }), 500

    if not os.path.exists(fixed_video):
        return jsonify({"error": "Fixed video not created"}), 500

    # ---------------- STEP 2: WHISPER ----------------
    try:
        result = model.transcribe(
            fixed_video,
            language=language
        )

        segments = result["segments"]

    except Exception as e:
        return jsonify({"error": f"Whisper failed: {str(e)}"}), 500

    # ---------------- STEP 3: CREATE SRT ----------------
    try:
        with open(subtitle_file, "w", encoding="utf-8") as srt:

            for i, segment in enumerate(segments, start=1):

                text = clean_text(segment["text"])

                if len(text) < 1:
                    continue

                start = format_srt_time(segment["start"])
                end = format_srt_time(segment["end"])

                srt.write(f"{i}\n")
                srt.write(f"{start} --> {end}\n")
                srt.write(f"{text}\n\n")

    except Exception as e:
        return jsonify({"error": f"SRT creation failed: {str(e)}"}), 500

    # ---------------- STEP 4: SAFE SUBTITLE PATH ----------------
    subtitle_path = os.path.abspath(subtitle_file)

    # WINDOWS SAFE PATH
    subtitle_path = subtitle_path.replace('\\', '/')
    subtitle_path = subtitle_path.replace(':', '\\:')

    # ---------------- STEP 5: BURN SUBTITLES ----------------
    burn_cmd = [
        "ffmpeg",
        "-y",
        "-i", fixed_video,
        "-vf", f"subtitles='{subtitle_path}'",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-c:a", "aac",
        "-pix_fmt", "yuv420p",
        final_video
    ]

    burn = subprocess.run(
        burn_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # DEBUG LOG
    print("FFMPEG STDERR:")
    print(burn.stderr)

    if burn.returncode != 0:
        return jsonify({
            "error": "Subtitle burn failed",
            "details": burn.stderr
        }), 500

    # ---------------- STEP 6: VALIDATE OUTPUT ----------------
    if not os.path.exists(final_video):
        return jsonify({"error": "Final video not created"}), 500

    size = os.path.getsize(final_video)

    if size < 100000:
        return jsonify({
            "error": "Output video too small",
            "size": size
        }), 500

    # ---------------- SEND FILE ----------------
    response = send_file(
        final_video,
        as_attachment=True
    )

    # ---------------- AUTO DELETE FILES ----------------
    threading.Thread(
        target=cleanup,
        args=([
            input_video,
            fixed_video,
            subtitle_file,
            final_video
        ],)
    ).start()

    return response


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)