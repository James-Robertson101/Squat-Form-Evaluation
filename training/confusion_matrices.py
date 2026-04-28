"""
confusion_matrices.py

Proper evaluation script:
- Splits data into train/test using GroupShuffleSplit
- Trains model ONLY on train split
- Evaluates on unseen test split
- Generates confusion matrices per label
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix

from save_models import (
    build_pipeline,
    remap_labels,
    FRONT_MODEL,
    SIDE_MODEL
)

# ── Paths ─────────────────────────────────────────────────────────
FRONT_CSV  = r"C:\Users\james\Squat Form Evaluation\datasets\front\front_view_merged.csv"
SIDE_CSV   = r"C:\Users\james\Squat Form Evaluation\datasets\side\side_view_merged.csv"
OUTPUT_DIR = "confusion_matrices"

# ── Features ──────────────────────────────────────────────────────
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

SIDE_FEATURES = [
    "hip_rom", "knee_rom", "torso_stability",
    "heel_instability", "toe_instability",
    "knee_min_angle", "hip_min_angle",
    "torso_lean_peak", "torso_lean_mean",
    "descent_frames", "ascent_frames", "descent_ascent_ratio",
    "knee_over_toe_mean", "knee_over_toe_max",
    "hip_below_knee_frac",
]

# ── Labels ────────────────────────────────────────────────────────
FRONT_LABELS = ["knee_valgus", "knee_varus", "lateral_hip_shift",
                "torso_lateral_lean", "foot_stability"]

SIDE_LABELS  = ["squat_depth", "lumbar_flexion", "forward_lean",
                "descent_control", "ascent_sticking", "foot_stability"]

# ── Display names ─────────────────────────────────────────────────
LABEL_CLASS_NAMES = {
    "knee_valgus":        {0: "Fine",        1: "Mild",       2: "Severe"},
    "knee_varus":         {0: "Fine",        1: "Mild",       2: "Severe"},
    "lateral_hip_shift":  {0: "Fine",        1: "Mild",       2: "Severe"},
    "torso_lateral_lean": {0: "Neutral",     1: "Mild",       2: "Excessive"},
    "foot_stability":     {0: "Stable",      1: "Mild",       2: "Significant"},
    "squat_depth":        {0: "Shallow",     1: "Sufficient", 2: "Deep"},
    "lumbar_flexion":     {0: "Neutral",     1: "Mild",       2: "Excessive"},
    "forward_lean":       {0: "Appropriate", 1: "Excessive"},
    "descent_control":    {0: "Controlled",  1: "Fast",       2: "Uncontrolled"},
    "ascent_sticking":    {0: "Smooth",      1: "Minor",      2: "Severe"},
}


# ── Main function ─────────────────────────────────────────────────
def plot_confusion_matrices(csv_path, feature_cols, label_cols, prefix, out_dir):

    print(f"\n{prefix.upper()} view")

    df = pd.read_csv(csv_path)

    groups = df['video_name'].values
    X = df[feature_cols]
    y = df[label_cols]

    # ── Train/test split ─────────────────────────────────────────
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train = X.iloc[train_idx]
    X_test  = X.iloc[test_idx]

    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test  = y.iloc[test_idx].reset_index(drop=True)

    print(f"  Train size: {len(X_train)}")
    print(f"  Test size : {len(X_test)}")

    # ── Remap labels ─────────────────────────────────────────────
    y_train_remap, encoders = remap_labels(y_train)

    classifier = FRONT_MODEL if prefix == "front" else SIDE_MODEL

    pipeline = build_pipeline(classifier)
    pipeline.fit(X_train, y_train_remap)

    # ── Predict ──────────────────────────────────────────────────
    y_pred_encoded = pipeline.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred_encoded, columns=label_cols)

    # ── Decode ───────────────────────────────────────────────────
    decoders = {col: {v: k for k, v in enc.items()} for col, enc in encoders.items()}

    y_pred_decoded = pd.DataFrame({
        col: y_pred_df[col].map(decoders[col])
        for col in label_cols
    })

    # ── Plotting (FIXED INDENTATION HERE) ────────────────────────
    os.makedirs(out_dir, exist_ok=True)

    n_labels = len(label_cols)
    fig, axes = plt.subplots(1, n_labels, figsize=(5 * n_labels, 5))
    if n_labels == 1:
        axes = [axes]

    fig.suptitle(f"{prefix.capitalize()} view — confusion matrices",
                 fontsize=14, fontweight='bold', y=1.05)

    print(f"\n{'LABEL':<25} {'CLASS DISTRIBUTION (TEST SET)'}")
    print("-" * 60)

    for ax, col in zip(axes, label_cols):

        y_true = list(y_test[col].values)
        y_pred = list(y_pred_decoded[col].values)

        # distribution
        counts = pd.Series(y_true).value_counts().sort_index()
        total = len(y_true)

        dist_str = " | ".join([
            f"{cls}:{count/total:.2f}"
            for cls, count in counts.items()
        ])

        print(f"{col:<25} {dist_str}")

        # confusion matrix
        classes = sorted(set(y_true) | set(y_pred))
        name_map = LABEL_CLASS_NAMES.get(col, {})
        class_labels = [name_map.get(c, str(c)) for c in classes]

        cm = confusion_matrix(y_true, y_pred, labels=classes)
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-6)

        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels,
            ax=ax,
            vmin=0,
            vmax=1,
            cbar=False
        )

        # counts
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(
                    j + 0.5, i + 0.75,
                    f"n={cm[i, j]}",
                    ha='center', va='center',
                    fontsize=7, color='grey'
                )

        ax.set_title(col.replace('_', ' ').title())
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()

    out_path = os.path.join(out_dir, f"{prefix}_confusion_matrices.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved → {out_path}")


# ── Run ───────────────────────────────────────────────────────────
if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plot_confusion_matrices(
        FRONT_CSV, FRONT_FEATURES, FRONT_LABELS, "front", OUTPUT_DIR
    )

    plot_confusion_matrices(
        SIDE_CSV, SIDE_FEATURES, SIDE_LABELS, "side", OUTPUT_DIR
    )

    print("\nDone.")