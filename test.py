from FeatureExtraction import extract_squat_features
import pandas as pd

# Separate lists for front and side features
front_features = []
side_features = []

videos = [
    ("Squat_Front_View.mp4", "front"),
    ("Squat_Side_View.mp4", "side")
]

for vid_path, view in videos:
    features = extract_squat_features(vid_path, view=view)
    if view == "front":
        front_features.extend(features)
    elif view == "side":
        side_features.extend(features)

# Save separate CSVs
if front_features:
    df_front = pd.DataFrame(front_features)
    df_front.to_csv("front_view_features.csv", index=False)
    print(f"Saved {len(front_features)} front view reps to front_view_features.csv")

if side_features:
    df_side = pd.DataFrame(side_features)
    df_side.to_csv("side_view_features.csv", index=False)
    print(f"Saved {len(side_features)} side view reps to side_view_features.csv")
