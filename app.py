import os
import re
import uuid
import shutil
from flask import Flask, request, jsonify, render_template, send_file, abort
from extract_frames import extract_frames
from prediction import predict_reps

app = Flask(__name__)

UPLOAD_FOLDER    = "uploads"
FRAMES_FOLDER    = "frames"
ANNOTATED_FOLDER = "annotated_frames"

for folder in (UPLOAD_FOLDER, FRAMES_FOLDER, ANNOTATED_FOLDER):
    os.makedirs(folder, exist_ok=True)

UUID_RE = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
)

def _valid_job_id(job_id: str) -> bool:
    return bool(UUID_RE.match(job_id or ""))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyse", methods=["POST"])
def analyse():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files["video"]
    view       = request.form.get("view", "side")

    if view not in ("side", "front"):
        return jsonify({"error": "Invalid view parameter"}), 400

    job_id           = str(uuid.uuid4())
    video_path       = os.path.join(UPLOAD_FOLDER,    f"{job_id}.mp4")
    frame_folder     = os.path.join(FRAMES_FOLDER,    job_id)
    annotated_folder = os.path.join(ANNOTATED_FOLDER, job_id)

    video_file.save(video_path)

    try:
        extract_frames(video_path, frame_folder)
        results = predict_reps(
            frame_folder,
            view          = view,
            annotated_out = annotated_folder,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(frame_folder):
            shutil.rmtree(frame_folder)

    if not results:
        if os.path.exists(annotated_folder):
            shutil.rmtree(annotated_folder)
        return jsonify({"error": "No reps detected in video"}), 400

    frame_count = 0
    if os.path.exists(annotated_folder):
        frame_count = len([
            f for f in os.listdir(annotated_folder)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])

    return jsonify({
        "view":        view,
        "reps":        results,
        "job_id":      job_id,
        "frame_count": frame_count,
    })


def _natural_key(s):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'([0-9]+)', s)]


@app.route("/frames/<job_id>/<int:frame_idx>")
def serve_frame(job_id, frame_idx):
    if not _valid_job_id(job_id):
        abort(400)

    folder = os.path.join(ANNOTATED_FOLDER, job_id)
    if not os.path.isdir(folder):
        abort(404)

    files = sorted(
        [f for f in os.listdir(folder)
         if f.lower().endswith((".jpg", ".jpeg", ".png"))],
        key=_natural_key,
    )

    if frame_idx < 0 or frame_idx >= len(files):
        abort(404)

    return send_file(
        os.path.join(folder, files[frame_idx]),
        mimetype="image/jpeg",
    )


@app.route("/cleanup/<job_id>", methods=["DELETE"])
def cleanup(job_id):
    if not _valid_job_id(job_id):
        abort(400)

    folder = os.path.join(ANNOTATED_FOLDER, job_id)
    if os.path.isdir(folder):
        shutil.rmtree(folder)

    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(debug=True)