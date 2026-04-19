import os
import uuid
import shutil
from flask import Flask, request, jsonify, render_template
from extract_frames import extract_frames
from prediction import predict_reps

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
FRAMES_FOLDER = "frames"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAMES_FOLDER, exist_ok=True)


@app.route("/") #default route
def index():
    return render_template("index.html")


@app.route("/analyse", methods=["POST"]) # api endpoint for feature extraction and prediction 
def analyse():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files["video"]
    view = request.form.get("view", "side")  # "side" or "front"

    # Save the uploaded video with a unique name to avoid collisions
    job_id      = str(uuid.uuid4())
    video_path  = os.path.join(UPLOAD_FOLDER, f"{job_id}.mp4")
    frame_folder = os.path.join(FRAMES_FOLDER, job_id)

    video_file.save(video_path)

    try:
        extract_frames(video_path, frame_folder)
        results = predict_reps(frame_folder, view=view)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # clean up — delete video and frames after processing
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(frame_folder):
            shutil.rmtree(frame_folder)

    if not results:
        return jsonify({"error": "No reps detected in video"}), 400

    return jsonify({"view": view, "reps": results})


if __name__ == "__main__":
    app.run(debug=True)