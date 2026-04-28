import joblib
import numpy as np
import pandas as pd
from FeatureExtraction import extract_squat_features_from_frames

# Feedback dictionaries
FEEDBACK_MAP = {
    "side": {
        "squat_depth": {
            2: {"status": "Below parallel", "severity": "good",  "detail": "Hip crease passes below the knee. Full range of motion achieved.", "cue": None},
            1: {"status": "Parallel",       "severity": "warn",  "detail": "Hip reaches roughly knee height. Try sitting back and down a little further.", "cue": "Drive your knees out and think 'sit into the floor' on the descent."},
            0: {"status": "Shallow",        "severity": "bad",   "detail": "Insufficient depth — hip stays well above the knee.", "cue": "Check ankle mobility. Elevate heels slightly or add a heel wedge to assist depth."},
        },
        "lumbar_flexion": {
            0: {"status": "Neutral spine",  "severity": "good",  "detail": "Lumbar curve maintained throughout the movement.", "cue": None},
            1: {"status": "Mild rounding",  "severity": "warn",  "detail": "Slight loss of lumbar lordosis near the bottom of the squat.", "cue": "Brace your core harder before descending. Think about 'proud chest' at the bottom."},
            2: {"status": "Butt wink",      "severity": "bad",   "detail": "Notable posterior pelvic tilt detected at depth — increases disc stress.", "cue": "Work on hip flexor and hamstring mobility. Reduce depth until neutral spine is maintained."},
        },
        "forward_lean": {
            0: {"status": "Upright",        "severity": "good", "detail": "Torso stays close to vertical throughout.", "cue": None},
            1: {"status": "Moderate lean",  "severity": "warn", "detail": "Some forward lean present — may be acceptable depending on build.", "cue": "Keep the bar path vertical. Drive elbows down if using a barbell."},
            2: {"status": "Excessive lean", "severity": "bad",  "detail": "Excessive forward lean detected — shifts load heavily to lower back.", "cue": "Strengthen upper back and improve ankle dorsiflexion. High-bar squat position may help."},
        },
        "descent_control": {
            0: {"status": "Controlled",     "severity": "good", "detail": "Smooth, controlled eccentric phase.", "cue": None},
            1: {"status": "Slightly fast",  "severity": "warn", "detail": "Descent is a little rushed — reduces time under tension.", "cue": "Aim for 2–3 seconds on the way down. Count in your head."},
            2: {"status": "Uncontrolled",   "severity": "bad",  "detail": "Rapid drop with minimal eccentric control.", "cue": "Significantly slow the descent. Consider reducing load until control improves."},
        },
        "ascent_sticking": {
            0: {"status": "Smooth ascent",        "severity": "good", "detail": "Consistent drive out of the hole with no sticking point.", "cue": None},
            1: {"status": "Minor sticking point", "severity": "warn", "detail": "Slight slowdown detected just above the bottom position.", "cue": "Pause squats and box squats can help strengthen this range."},
            2: {"status": "Significant sticking", "severity": "bad",  "detail": "Noticeable stall on the way up — suggests weakness at that range.", "cue": "Add volume work at that range. Pin squats or tempo squats target the sticking point directly."},
        },
        "foot_stability": {
            0: {"status": "Stable",     "severity": "good", "detail": "Feet remain flat and grounded throughout.", "cue": None},
            1: {"status": "Minor lift", "severity": "warn", "detail": "Slight heel or toe instability detected.", "cue": "Press through the full foot — big toe, little toe, heel. Think of a tripod."},
            2: {"status": "Unstable",   "severity": "bad",  "detail": "Heel rising or toe lifting significantly disrupts force transfer.", "cue": "Work on ankle mobility. Raised-heel squats can be a short-term fix while you address the root cause."},
        },
    },
    "front": {
        "knee_valgus": {
            0: {"status": "No valgus",          "severity": "good", "detail": "Knees track well over toes throughout the movement.", "cue": None},
            1: {"status": "Mild valgus",         "severity": "warn", "detail": "Knees drift slightly inward, especially near the bottom.", "cue": "Cue 'spread the floor' with your feet. Strengthen glute med with banded walks."},
            2: {"status": "Significant valgus",  "severity": "bad",  "detail": "Knees collapse noticeably inward — high knee stress.", "cue": "Reduce load immediately. Use a light resistance band above the knees as proprioceptive feedback."},
        },
        "knee_varus": {
            0: {"status": "No varus",           "severity": "good", "detail": "Knees stay aligned over toes.", "cue": None},
            1: {"status": "Mild varus",          "severity": "warn", "detail": "Knees flare outward slightly past the toes.", "cue": "Check foot angle — too wide a stance can cause this. Try narrowing slightly."},
            2: {"status": "Significant varus",   "severity": "bad",  "detail": "Excessive knee flare detected.", "cue": "Reassess stance width and toe angle. Improve adductor flexibility and control."},
        },
        "lateral_hip_shift": {
            0: {"status": "Symmetric",          "severity": "good", "detail": "Hips stay centred throughout the squat.", "cue": None},
            1: {"status": "Mild shift",          "severity": "warn", "detail": "Hips drift slightly to one side.", "cue": "Focus on equal loading through both feet. Single-leg work can address asymmetry."},
            2: {"status": "Significant shift",   "severity": "bad",  "detail": "Clear lateral hip drift detected — suggests mobility or strength imbalance.", "cue": "Investigate hip mobility asymmetry. Bulgarian split squats can help balance sides."},
        },
        "torso_lateral_lean": {
            0: {"status": "Centred",            "severity": "good", "detail": "Torso remains stacked over hips.", "cue": None},
            1: {"status": "Mild lean",           "severity": "warn", "detail": "Torso leans slightly to one side.", "cue": "Check for shoulder or hip asymmetry. Core anti-lateral flexion work (e.g. Pallof press) can help."},
            2: {"status": "Excessive lean",      "severity": "bad",  "detail": "Torso tilts significantly — places uneven load on the spine.", "cue": "Reduce load and assess for hip or thoracic mobility restrictions."},
        },
        "foot_stability": {
            0: {"status": "Stable",             "severity": "good", "detail": "Stance width consistent and feet grounded.", "cue": None},
            1: {"status": "Minor instability",   "severity": "warn", "detail": "Slight variation in stance or foot position.", "cue": "Mark foot placement before sets. Focus on consistent setup."},
            2: {"status": "Unstable",            "severity": "bad",  "detail": "Feet shift or stance width varies significantly between reps.", "cue": "Use tape markers on the floor to establish a repeatable stance. Slow down the setup."},
        },
    },
}


def load_model_bundle(prefix, models_dir="models"):
    return {
        "model":        joblib.load(f"{models_dir}/{prefix}_model.pkl"),
        "imputer":      joblib.load(f"{models_dir}/{prefix}_imputer.pkl"),
        "scaler":       joblib.load(f"{models_dir}/{prefix}_scaler.pkl"),
        "feature_cols": joblib.load(f"{models_dir}/{prefix}_feature_cols.pkl"),
    }


def predict_reps(frame_folder, view="side", models_dir="models", annotated_out=None):
    """
    Runs the full pipeline on a folder of frames and returns feedback dicts
    for each detected rep, ready to hand straight to the frontend.

    annotated_out: folder to write keypoint-overlay frames into (used by the scrubber).
    """
    per_rep_features = extract_squat_features_from_frames(
        frame_folder,
        view          = view,
        annotated_out = annotated_out,
    )

    if not per_rep_features:
        return []

    bundle = load_model_bundle(view, models_dir)

    X = pd.DataFrame(per_rep_features)[bundle["feature_cols"]]

    X_imputed = bundle["imputer"].transform(X)
    X_scaled  = bundle["scaler"].transform(X_imputed)

    raw_predictions = bundle["model"].predict(X_scaled)

    label_cols   = list(FEEDBACK_MAP[view].keys())
    decoded_reps = []

    for rep_idx, pred_row in enumerate(raw_predictions):
        rep_result = {"rep": rep_idx + 1, "labels": {}}
        for col, pred_int in zip(label_cols, pred_row):
            rep_result["labels"][col] = FEEDBACK_MAP[view][col][int(pred_int)]
        decoded_reps.append(rep_result)

    return decoded_reps