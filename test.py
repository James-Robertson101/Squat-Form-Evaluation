"""
test.py
Covers UT-01 to UT-08, IT-01 to IT-10, PT-01 to PT-03.
Run with:  pytest test.py -v
"""
import sys
import os
from unittest.mock import MagicMock
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "training"))

sys.modules["mediapipe"]                          = MagicMock()
sys.modules["mediapipe.solutions"]                = MagicMock()
sys.modules["mediapipe.solutions.pose"]           = MagicMock()
sys.modules["mediapipe.solutions.drawing_utils"]  = MagicMock()
sys.modules["mediapipe.solutions.drawing_styles"] = MagicMock()
sys.modules["Remapper"]                           = MagicMock()
sys.modules["FeatureExtraction"]                  = MagicMock()
sys.modules["prediction"]                         = MagicMock()

import io
import json
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from sklearn.model_selection import GroupShuffleSplit

from app import app
from training.hyperparameter_tuning import remap_labels, multioutput_f1, exact_match, hamming

# Helpers
def fake_video():
    return (io.BytesIO(b"fake video content"), "squat.mp4")

MOCK_RESULTS = [{
    "rep": 1,
    "labels": {
        "squat_depth":     {"status": "Below parallel", "severity": "good", "detail": "...", "cue": None},
        "lumbar_flexion":  {"status": "Neutral spine",  "severity": "good", "detail": "...", "cue": None},
        "forward_lean":    {"status": "Upright",        "severity": "good", "detail": "...", "cue": None},
        "descent_control": {"status": "Controlled",     "severity": "good", "detail": "...", "cue": None},
        "ascent_sticking": {"status": "Smooth ascent",  "severity": "good", "detail": "...", "cue": None},
        "foot_stability":  {"status": "Stable",         "severity": "good", "detail": "...", "cue": None},
    }
}]

@pytest.fixture
def client():
    app.config["TESTING"] = True
    return app.test_client()
# UT-01  Empty frame folder returns []
def test_UT01_empty_folder_returns_empty(tmp_path):
    del sys.modules["FeatureExtraction"]
    try:
        from FeatureExtraction import extract_squat_features_from_frames
        assert extract_squat_features_from_frames(str(tmp_path), view="side") == []
    finally:
        sys.modules["FeatureExtraction"] = MagicMock()

# UT-02  Blank images with no detectable pose return []
def test_UT02_blank_frames_return_empty(tmp_path):
    import cv2
    del sys.modules["FeatureExtraction"]
    try:
        from FeatureExtraction import extract_squat_features_from_frames
        for i in range(3):
            cv2.imwrite(str(tmp_path / f"frame_{i}.jpg"),
                        np.zeros((480, 640, 3), dtype=np.uint8))
        assert extract_squat_features_from_frames(str(tmp_path), view="side") == []
    finally:
        sys.modules["FeatureExtraction"] = MagicMock()


# UT-03  remap_labels maps sorted strings to 0-based integers
def test_UT03_remap_labels_basic():
    y_train = pd.DataFrame({"label": ["bad", "good", "warn"]})
    y_test  = pd.DataFrame({"label": ["good", "bad"]})
    y_tr, y_te = remap_labels(y_train, y_test)
    assert list(y_tr["label"]) == [0, 1, 2]   # bad→0, good→1, warn→2
    assert list(y_te["label"]) == [1, 0]

# UT-04  Unseen test label maps to -1
def test_UT04_unseen_label_maps_to_minus_one():
    y_train = pd.DataFrame({"label": ["good", "bad"]})
    y_test  = pd.DataFrame({"label": ["unseen"]})
    _, y_te = remap_labels(y_train, y_test)
    assert y_te["label"].iloc[0] == -1


# UT-05  multioutput_f1 returns 1.0 on identical arrays
def test_UT05_perfect_f1():
    y = np.array([[0, 1], [1, 0], [2, 1]])
    assert multioutput_f1(y, y) == pytest.approx(1.0)

# UT-06  exact_match returns 0.0 when all predictions wrong
def test_UT06_zero_exact_match_on_all_wrong():
    assert exact_match(np.array([[0, 0]]), np.array([[1, 1]])) == pytest.approx(0.0)

# UT-07  hamming returns 0.0 on perfect predictions
def test_UT07_hamming_zero_on_perfect():
    y = np.array([[0, 1], [1, 0]])
    assert hamming(y, y) == pytest.approx(0.0)


# UT-08  hamming returns 1.0 when all labels wrong
def test_UT08_hamming_one_on_all_wrong():
    assert hamming(np.array([[0, 0]]), np.array([[1, 1]])) == pytest.approx(1.0)

# IT-01  /analyse with no video returns 400
def test_IT01_analyse_missing_video_returns_400(client):
    r = client.post("/analyse", data={"view": "side"})
    assert r.status_code == 400

# IT-02  /analyse with invalid view returns 400
def test_IT02_analyse_invalid_view_returns_400(client):
    r = client.post("/analyse",
                    data={"video": fake_video(), "view": "diagonal"},
                    content_type="multipart/form-data")
    assert r.status_code == 400

# IT-03  /analyse with valid video and mocked pipeline returns 200
def test_IT03_analyse_valid_upload_returns_200(client):
    with patch("app.extract_frames"), \
         patch("app.predict_reps", return_value=MOCK_RESULTS):
        r = client.post("/analyse",
                        data={"video": fake_video(), "view": "side"},
                        content_type="multipart/form-data")
    assert r.status_code == 200
    body = json.loads(r.data)
    assert body["view"] == "side"
    assert len(body["reps"]) == 1
    assert "job_id" in body

# IT-04  /analyse when no reps detected returns 400
def test_IT04_analyse_no_reps_returns_400(client):
    with patch("app.extract_frames"), \
         patch("app.predict_reps", return_value=[]):
        r = client.post("/analyse",
                        data={"video": fake_video(), "view": "side"},
                        content_type="multipart/form-data")
    assert r.status_code == 400

# IT-05  /frames with malformed job ID returns 400
def test_IT05_frames_invalid_job_id_returns_400(client):
    r = client.get("/frames/not-a-uuid/0")
    assert r.status_code == 400

# IT-06  /frames with unknown job ID returns 404
def test_IT06_frames_missing_job_returns_404(client):
    r = client.get("/frames/12345678-1234-1234-1234-123456789abc/0")
    assert r.status_code == 404

# IT-07  /cleanup with malformed job ID returns 400
def test_IT07_cleanup_invalid_job_id_returns_400(client):
    r = client.post("/cleanup/not-a-uuid")
    assert r.status_code == 400

# IT-08  /cleanup removes annotated frame directory
def test_IT08_cleanup_removes_folder(client, tmp_path, monkeypatch):
    import app as app_module
    job_id = "12345678-1234-1234-1234-123456789abc"
    (tmp_path / job_id).mkdir()
    monkeypatch.setattr(app_module, "ANNOTATED_FOLDER", str(tmp_path))
    r = client.post(f"/cleanup/{job_id}")
    assert r.status_code == 200
    assert not (tmp_path / job_id).exists()

# IT-09  predict_reps returns [] when no reps extracted
def test_IT09_predict_reps_empty_when_no_features():
    del sys.modules["prediction"]
    try:
        with patch("prediction.extract_squat_features_from_frames", return_value=[]):
            from prediction import predict_reps
            assert predict_reps("any_folder", view="side") == []
    finally:
        sys.modules["prediction"] = MagicMock()

# IT-10  Each rep result contains rep, labels, and all expected label keys
def test_IT10_rep_result_has_correct_structure():
    dummy_features = [{
        "video_name": "test", "hip_rom": 45.0, "knee_rom": 80.0,
        "torso_stability": 3.2, "heel_instability": 0.5, "toe_instability": 0.3,
        "knee_min_angle": 85.0, "hip_min_angle": 70.0, "torso_lean_peak": 25.0,
        "torso_lean_mean": 18.0, "descent_frames": 20.0, "ascent_frames": 15.0,
        "descent_ascent_ratio": 1.3, "knee_over_toe_mean": 0.1,
        "knee_over_toe_max": 0.2, "hip_below_knee_frac": 0.8,
    }]
    feature_cols = [k for k in dummy_features[0] if k != "video_name"]

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0, 0, 0, 0, 0, 0]])
    mock_bundle = {
        "model":        mock_model,
        "imputer":      MagicMock(transform=lambda x: x),
        "scaler":       MagicMock(transform=lambda x: x),
        "feature_cols": feature_cols,
    }

    del sys.modules["prediction"]
    try:
        with patch("prediction.extract_squat_features_from_frames", return_value=dummy_features), \
             patch("prediction.load_model_bundle", return_value=mock_bundle):
            from prediction import predict_reps, FEEDBACK_MAP
            results = predict_reps("any_folder", view="side")

        assert len(results) == 1
        rep = results[0]
        assert "rep"    in rep
        assert "labels" in rep
        for key in FEEDBACK_MAP["side"]:
            assert key in rep["labels"], f"Missing label key: {key}"
    finally:
        sys.modules["prediction"] = MagicMock()

# PT-01  No video appears in both train and test splits (no data leakage)
def test_PT01_no_video_overlap_in_group_split():
    groups = np.repeat([f"video_{i}" for i in range(20)], 5)
    X      = pd.DataFrame(np.random.rand(100, 5))
    gss    = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, groups=groups))
    overlap = set(groups[train_idx]) & set(groups[test_idx])
    assert len(overlap) == 0, f"Data leakage — shared videos: {overlap}"

# PT-02  Front view KNN achieves >= 0.70 weighted F1 on held-out test set
def test_PT02_front_view_f1_above_threshold():
    import joblib

    FRONT_CSV = r"C:\Users\james\Squat Form Evaluation\datasets\front\front_view_merged.csv"
    FRONT_FEATURES = [
        "valgus_min", "valgus_max", "valgus_variation",
        "torso_lateral_peak", "symmetry_mean",
        "heel_wobble", "heel_instability", "toe_instability",
        "knee_cave_frames", "knee_cave_frac",
        "knee_asym_mean", "knee_asym_std",
        "ankle_width_mean", "ankle_width_std",
        "hip_shift_max", "hip_shift_std",
        "sho_hip_offset_mean", "sho_hip_offset_max",
    ]
    FRONT_LABELS = ["knee_valgus", "knee_varus", "lateral_hip_shift",
                    "torso_lateral_lean", "foot_stability"]

    df     = pd.read_csv(FRONT_CSV)
    groups = df["video_name"].values
    X, y   = df[FRONT_FEATURES], df[FRONT_LABELS]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, test_idx = next(gss.split(X, y, groups=groups))

    pipeline = joblib.load("models/front_pipeline.pkl")
    score    = multioutput_f1(y.iloc[test_idx].values, pipeline.predict(X.iloc[test_idx]))
    assert score >= 0.70, f"Front view F1 {score:.3f} below 0.70 threshold"

# PT-03  Side view XGBoost achieves >= 0.79 weighted F1 on held-out test set
def test_PT03_side_view_f1_above_threshold():
    import joblib

    SIDE_CSV = r"C:\Users\james\Squat Form Evaluation\datasets\side\side_view_merged.csv"
    SIDE_FEATURES = [
        "hip_rom", "knee_rom", "torso_stability",
        "heel_instability", "toe_instability",
        "knee_min_angle", "hip_min_angle",
        "torso_lean_peak", "torso_lean_mean",
        "descent_frames", "ascent_frames", "descent_ascent_ratio",
        "knee_over_toe_mean", "knee_over_toe_max",
        "hip_below_knee_frac",
    ]
    SIDE_LABELS = ["squat_depth", "lumbar_flexion", "forward_lean",
                   "descent_control", "ascent_sticking", "foot_stability"]

    df     = pd.read_csv(SIDE_CSV)
    groups = df["video_name"].values
    X, y   = df[SIDE_FEATURES], df[SIDE_LABELS]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, test_idx = next(gss.split(X, y, groups=groups))

    pipeline = joblib.load("models/side_pipeline.pkl")
    score    = multioutput_f1(y.iloc[test_idx].values, pipeline.predict(X.iloc[test_idx]))
    assert score >= 0.79, f"Side view F1 {score:.3f} below 0.79 threshold"