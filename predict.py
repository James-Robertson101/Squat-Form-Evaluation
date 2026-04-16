# predict.py
import joblib
import numpy as np
import pandas as pd
import tempfile
import os
import cv2
from FeatureExtraction import extract_squat_features_from_frames

# Label descriptions for human-readable feedback
LABEL_DESCRIPTIONS = {
    # Front
    "knee_valgus":          {0: "No knee cave",         1: "Mild knee cave",      2: "Severe knee cave"},
    "knee_varus":           {0: "No bow-legged",        1: "Mild bow-legged"},
    "lateral_hip_shift":    {0: "Hips centred",         1: "Mild hip shift",      2: "Severe hip shift"},
    "torso_lateral_lean":   {0: "Torso upright",        1: "Mild lateral lean",   2: "Severe lateral lean"},
    "foot_stability":       {0: "Feet stable",          1: "Minor instability",   2: "Unstable feet"},
    # Side
    "squat_depth":          {0: "Below parallel",       1: "At parallel",         2: "Above parallel (shallow)"},
    "lumbar_flexion":       {0: "Neutral spine",        1: "Mild rounding",       2: "Excessive rounding"},
    "forward_lean":         {0: "Good upright torso",   1: "Moderate lean",       2: "Excessive forward lean"},
    "descent_control":      {0: "Controlled descent",   1: "Slightly fast",       2: "Uncontrolled drop"},
    "ascent_sticking":      {0: "Smooth ascent",        1: "Minor sticking point",2: "Clear sticking point"},
}

def load_artifacts(view: str) -> dict:
    return {
        "model":    joblib.load(f"models/{view}_model.pkl"),
        "imputer":  joblib.load(f"models/{view}_imputer.pkl"),
        "scaler":   joblib.load(f"models/{view}_scaler.pkl"),
        "encoders": joblib.load(f"models/{view}_encoders.pkl"),
    }

def video_to_frames(video_path: str, out_dir: str, max_frames: int = 300) -> int:
    """Extract frames from a video file into out_dir. Returns frame count."""
    cap = cv2.VideoCapture(video_path)
    count = 0
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(out_dir, f"frame_{count:05d}.jpg"), frame)
        count += 1
    cap.release()
    return count

def predict_video(video_path: str, view: str) -> dict:
    """
    Run the full pipeline on a video file.

    Returns a dict with:
      - 'reps': list of per-rep predictions
      - 'summary': aggregated label predictions across all reps
      - 'feedback': human-readable feedback strings
    """
    artifacts = load_artifacts(view)
    model, imputer, scaler, encoders = (
        artifacts["model"], artifacts["imputer"],
        artifacts["scaler"], artifacts["encoders"]
    )

    # Invert encoders: index → original label value
    decoders = {col: {v: k for k, v in enc.items()} for col, enc in encoders.items()}

    with tempfile.TemporaryDirectory() as tmpdir:
        frame_count = video_to_frames(video_path, tmpdir)
        if frame_count == 0:
            return {"error": "Could not read video frames"}

        raw_features = extract_squat_features_from_frames(tmpdir, view=view)

    if not raw_features:
        return {"error": "No reps detected — ensure the full squat is visible"}

    results = []
    for rep in raw_features:
        rep_copy = {k: v for k, v in rep.items() if k != "video_name"}
        X = pd.DataFrame([rep_copy])
        X_sc = scaler.transform(imputer.transform(X))
        pred = model.predict(X_sc)[0]

        label_cols = list(encoders.keys())
        decoded = {
            col: decoders[col].get(int(pred[i]), int(pred[i]))
            for i, col in enumerate(label_cols)
        }
        results.append(decoded)

    # Aggregate: take the worst (max) label per category across reps
    summary = {}
    feedback = []
    label_cols = list(encoders.keys())
    for col in label_cols:
        worst = max(r[col] for r in results)
        summary[col] = worst
        desc = LABEL_DESCRIPTIONS.get(col, {}).get(worst, str(worst))
        if worst > 0:
            feedback.append(f"⚠️  {col.replace('_', ' ').title()}: {desc}")
        else:
            feedback.append(f"✅  {col.replace('_', ' ').title()}: {desc}")

    return {
        "reps":     results,
        "summary":  summary,
        "feedback": feedback,
        "rep_count": len(results),
    }