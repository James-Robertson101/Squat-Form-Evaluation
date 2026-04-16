import pandas as pd

# Load CSVs
features = pd.read_csv("side_view_features.csv")
labels = pd.read_csv("side_view_labels.csv")

# Ensure video_name is numeric
features["video_name"] = pd.to_numeric(features["video_name"], errors="coerce")
labels["video_name"] = pd.to_numeric(labels["video_name"], errors="coerce")

# Sort by video_name
features = features.sort_values(by="video_name").reset_index(drop=True)
# output sorted features csv
features.to_csv("sorted_features.csv", index=False)

labels = labels.sort_values(by="video_name").reset_index(drop=True)

# Check for missing or extra videos
features_videos = set(features["video_name"])
labels_videos = set(labels["video_name"])

missing_in_features = labels_videos - features_videos
missing_in_labels = features_videos - labels_videos

if missing_in_features:
    print(f"Warning: these videos are in labels but not in features: {missing_in_features}")
if missing_in_labels:
    print(f"Warning: these videos are in features but not in labels: {missing_in_labels}")

# Merge on video_name
merged = pd.merge(features, labels, on="video_name", how="inner")

# Reorder cols feaures then lables
feature_cols = list(features.columns)
label_cols = [col for col in labels.columns if col != "video_name"]
merged = merged[["video_name"] + feature_cols[1:] + label_cols]

# Save merged CSV
merged.to_csv("side_view_merged.csv", index=False)

print("✅ Merge complete! Saved as 'side_view_merged.csv'.")