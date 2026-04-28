import cv2
import numpy as np
import os
import re
from scipy.signal import find_peaks, savgol_filter
from modules.calculateAngle import calculate_angle
from modules.midpoint import midpoint
from modules.smoothAngle import smooth_angle
import mediapipe as mp

# Mediapipe setup
mp_pose_module        = mp.solutions.pose
mp_pose_class         = mp_pose_module.Pose
mp_drawing            = mp.solutions.drawing_utils
mp_drawing_styles     = mp.solutions.drawing_styles

# Landmark indices
LH         = mp_pose_module.PoseLandmark.LEFT_HIP.value
RH         = mp_pose_module.PoseLandmark.RIGHT_HIP.value
LS         = mp_pose_module.PoseLandmark.LEFT_SHOULDER.value
RS         = mp_pose_module.PoseLandmark.RIGHT_SHOULDER.value
LK         = mp_pose_module.PoseLandmark.LEFT_KNEE.value
RK         = mp_pose_module.PoseLandmark.RIGHT_KNEE.value
LA         = mp_pose_module.PoseLandmark.LEFT_ANKLE.value
RA         = mp_pose_module.PoseLandmark.RIGHT_ANKLE.value
LE         = mp_pose_module.PoseLandmark.LEFT_EAR.value
RE         = mp_pose_module.PoseLandmark.RIGHT_EAR.value
L_HEEL     = mp_pose_module.PoseLandmark.LEFT_HEEL.value
R_HEEL     = mp_pose_module.PoseLandmark.RIGHT_HEEL.value
L_FOOT_IDX = mp_pose_module.PoseLandmark.LEFT_FOOT_INDEX.value
R_FOOT_IDX = mp_pose_module.PoseLandmark.RIGHT_FOOT_INDEX.value

VIS_THRESHOLD = 0.5

# Custom drawing spec colours ---------------------------------------------------
_LANDMARK_STYLE = mp_drawing.DrawingSpec(
    color=(0, 255, 180), thickness=2, circle_radius=3
)
_CONNECTION_STYLE = mp_drawing.DrawingSpec(
    color=(0, 200, 255), thickness=2
)


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'([0-9]+)', s)]


def landmarks_visible(landmarks, indices, threshold=VIS_THRESHOLD):
    return all(landmarks[i].visibility >= threshold for i in indices)


def build_rep_signal(landmarks_per_frame, view):
    hip_y_values = []
    for lm in landmarks_per_frame:
        if lm is None:
            continue
        hip_y = (lm[LH].y + lm[RH].y) / 2.0
        hip_y_values.append(hip_y)
    return np.array(hip_y_values)


def compute_descent_ascent(signal, bottom_idx, search_radius):
    start = max(0, bottom_idx - search_radius)
    end   = min(len(signal), bottom_idx + search_radius)

    descent_start = bottom_idx
    for i in range(bottom_idx, start - 1, -1):
        if signal[i] <= signal[descent_start]:
            descent_start = i
        else:
            break

    ascent_end = bottom_idx
    for i in range(bottom_idx, end):
        if signal[i] <= signal[ascent_end]:
            ascent_end = i
        else:
            break

    descent_frames = bottom_idx - descent_start
    ascent_frames  = ascent_end - bottom_idx
    return descent_frames, ascent_frames


def extract_squat_features_from_frames(
    frame_folder,
    view          = "side",
    debug         = False,
    annotated_out = None,       # folder to write keypoint-annotated frames into
):
    video_name = os.path.basename(frame_folder.rstrip("/\\"))

    if annotated_out:
        os.makedirs(annotated_out, exist_ok=True)

    pose = mp_pose_class(
        static_image_mode        = False,
        model_complexity         = 1,
        enable_segmentation      = False,
        min_detection_confidence = 0.05,
        min_tracking_confidence  = 0.05,
    )

    frame_files = sorted(
        [f for f in os.listdir(frame_folder)
         if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
        key=natural_sort_key,
    )

    fps = 30

    all_landmarks       = []
    valid_frame_indices = []

    hip_angles, knee_angles, torso_angles            = [], [], []
    heel_displacements, toe_displacements, heel_lateral = [], [], []

    torso_lean_angles = []
    knee_over_toe     = []
    hip_below_knee_arr = []

    valgus_ratios, lateral_lean, symmetry            = [], [], []
    left_knee_x_arr,  right_knee_x_arr               = [], []
    left_ankle_x_arr, right_ankle_x_arr              = [], []
    hip_center_x_arr                                  = []
    shoulder_center_x_arr                             = []

    prev_knee = prev_hip = prev_torso = None

    for frame_idx, frame_file in enumerate(frame_files):
        frame_path = os.path.join(frame_folder, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            all_landmarks.append(None)
            continue

        frame     = cv2.resize(frame, (640, 480))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = pose.process(frame_rgb)

        # --- annotate and save regardless of visibility -------------------------
        if annotated_out is not None:
            annotated = frame.copy()
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    annotated,
                    results.pose_landmarks,
                    mp_pose_module.POSE_CONNECTIONS,
                    landmark_drawing_spec = _LANDMARK_STYLE,
                    connection_drawing_spec = _CONNECTION_STYLE,
                )
            cv2.imwrite(os.path.join(annotated_out, frame_file), annotated)
        # ------------------------------------------------------------------------

        if not results.pose_landmarks:
            all_landmarks.append(None)
            continue

        lm = results.pose_landmarks.landmark

        core = [LH, RH, LS, RS]
        if not landmarks_visible(lm, core):
            all_landmarks.append(None)
            continue

        all_landmarks.append(lm)
        valid_frame_indices.append(frame_idx)

        h, w = 480, 640

        left_hip      = np.array([lm[LH].x * w, lm[LH].y * h])
        right_hip     = np.array([lm[RH].x * w, lm[RH].y * h])
        hip           = np.array(midpoint(left_hip, right_hip))

        left_shoulder  = np.array([lm[LS].x * w, lm[LS].y * h])
        right_shoulder = np.array([lm[RS].x * w, lm[RS].y * h])
        shoulder       = np.array(midpoint(left_shoulder, right_shoulder))

        left_ear  = lm[LE]
        right_ear = lm[RE]
        if left_ear.visibility >= VIS_THRESHOLD and right_ear.visibility >= VIS_THRESHOLD:
            torso_top = np.array([
                (left_ear.x + right_ear.x) / 2 * w,
                (left_ear.y + right_ear.y) / 2 * h,
            ])
        elif left_ear.visibility >= VIS_THRESHOLD:
            torso_top = np.array([left_ear.x * w, left_ear.y * h])
        elif right_ear.visibility >= VIS_THRESHOLD:
            torso_top = np.array([right_ear.x * w, right_ear.y * h])
        else:
            torso_top = shoulder

        left_heel  = np.array([lm[L_HEEL].x * w, lm[L_HEEL].y * h])
        right_heel = np.array([lm[R_HEEL].x * w, lm[R_HEEL].y * h])
        left_toe   = np.array([lm[L_FOOT_IDX].x * w, lm[L_FOOT_IDX].y * h])
        right_toe  = np.array([lm[R_FOOT_IDX].x * w, lm[R_FOOT_IDX].y * h])

        left_ankle_y  = lm[LA].y * h
        right_ankle_y = lm[RA].y * h

        heel_displacements.append(np.mean([
            abs(left_heel[1]  - left_ankle_y),
            abs(right_heel[1] - right_ankle_y),
        ]))
        toe_displacements.append(np.mean([
            abs(left_toe[1]  - left_ankle_y),
            abs(right_toe[1] - right_ankle_y),
        ]))
        heel_lateral.append(abs(left_heel[0] - right_heel[0]))

        # SIDE VIEW
        if view == "side":
            side_required = [LK, LA]
            if landmarks_visible(lm, side_required):
                knee  = np.array([lm[LK].x * w, lm[LK].y * h])
                ankle = np.array([lm[LA].x * w, lm[LA].y * h])

                knee_angle  = calculate_angle(hip, knee, ankle)
                hip_angle   = calculate_angle(shoulder, hip, knee)
                torso_angle = calculate_angle(hip, shoulder, torso_top)

                smoothed_knee  = smooth_angle(knee_angle,  prev_knee)
                smoothed_hip   = smooth_angle(hip_angle,   prev_hip)
                smoothed_torso = smooth_angle(torso_angle, prev_torso)

                prev_knee, prev_hip, prev_torso = smoothed_knee, smoothed_hip, smoothed_torso

                hip_angles.append(smoothed_hip)
                knee_angles.append(smoothed_knee)
                torso_angles.append(smoothed_torso)

                torso_vec = shoulder - hip
                vertical  = np.array([0, -1])
                cos_theta = np.dot(torso_vec, vertical) / (np.linalg.norm(torso_vec) + 1e-6)
                lean_deg  = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
                torso_lean_angles.append(lean_deg)

                thigh_len         = np.linalg.norm(hip - knee) + 1e-6
                knee_past_ankle_x = (knee[0] - ankle[0]) / thigh_len
                knee_over_toe.append(knee_past_ankle_x)

                hip_below_knee_arr.append(1.0 if hip[1] > knee[1] else 0.0)
            else:
                hip_angles.append(np.nan)
                knee_angles.append(np.nan)
                torso_angles.append(np.nan)
                torso_lean_angles.append(np.nan)
                knee_over_toe.append(np.nan)
                hip_below_knee_arr.append(np.nan)

        # FRONT VIEW
        elif view == "front":
            hip_center_x_arr.append(hip[0])
            shoulder_center_x_arr.append(shoulder[0])

            front_required = [LK, RK]
            if landmarks_visible(lm, front_required):
                left_knee  = np.array([lm[LK].x * w, lm[LK].y * h])
                right_knee = np.array([lm[RK].x * w, lm[RK].y * h])

                left_ankle_x  = lm[LA].x * w
                right_ankle_x = lm[RA].x * w

                hip_width  = abs(left_hip[0]  - right_hip[0])
                knee_width = abs(left_knee[0] - right_knee[0])

                valgus_ratio = knee_width / (hip_width + 1e-6)
                valgus_ratios.append(valgus_ratio)

                torso_mid = midpoint(left_shoulder, right_shoulder)
                hip_mid   = midpoint(left_hip, right_hip)
                lateral_lean.append(abs(torso_mid[0] - hip_mid[0]))
                symmetry.append(abs(left_shoulder[0] - right_shoulder[0]) / (hip_width + 1e-6))

                left_knee_x_arr.append(left_knee[0])
                right_knee_x_arr.append(right_knee[0])
                left_ankle_x_arr.append(left_ankle_x)
                right_ankle_x_arr.append(right_ankle_x)
            else:
                valgus_ratios.append(np.nan)
                lateral_lean.append(np.nan)
                symmetry.append(np.nan)
                left_knee_x_arr.append(np.nan)
                right_knee_x_arr.append(np.nan)
                left_ankle_x_arr.append(np.nan)
                right_ankle_x_arr.append(np.nan)

    pose.close()

    valid_count = sum(1 for l in all_landmarks if l is not None)
    if valid_count < 10:
        if debug:
            print(f"[FAIL] {video_name}: too few valid frames ({valid_count})")
        return []

    valid_landmarks = [l for l in all_landmarks if l is not None]
    raw_signal = build_rep_signal(valid_landmarks, view)

    if len(raw_signal) < 10:
        return []

    p5, p95     = np.percentile(raw_signal, 5), np.percentile(raw_signal, 95)
    raw_signal  = np.clip(raw_signal, p5, p95)

    win            = min(11, len(raw_signal) // 2 * 2 + 1)
    signal_smooth  = savgol_filter(raw_signal, win, 2)
    signal_for_peaks = signal_smooth
    signal_range   = np.ptp(signal_for_peaks)

    min_distance   = max(int(fps * 0.8), 5)
    min_prominence = signal_range * 0.20

    bottom_indices, _ = find_peaks(
        signal_for_peaks,
        distance   = min_distance,
        prominence = min_prominence,
    )

    if len(bottom_indices) > 1:
        merged = [bottom_indices[0]]
        for idx in bottom_indices[1:]:
            if idx - merged[-1] > min_distance:
                merged.append(idx)
            else:
                i_prev = merged[-1]
                if signal_for_peaks[idx] > signal_for_peaks[i_prev]:
                    merged[-1] = idx
        bottom_indices = np.array(merged)

    if len(bottom_indices) == 0 and signal_range > 0.02:
        bottom_indices = np.array([np.argmax(signal_for_peaks)])

    if debug:
        print(f"{video_name}: bottoms={len(bottom_indices)}, "
              f"signal_range={signal_range:.4f}, "
              f"min_prominence={min_prominence:.4f}")

    half_win     = int(fps * 0.6)
    n            = len(signal_for_peaks)
    tempo_radius = int(fps * 2.0)

    per_rep_features = []

    for idx in bottom_indices:
        start = max(0, idx - half_win)
        end   = min(n, idx + half_win)

        if end - start < 5:
            continue

        def safe_slice(arr, s, e):
            sliced = np.array(arr[s:e], dtype=float)
            return sliced[~np.isnan(sliced)]

        foot_heel = safe_slice(heel_displacements, start, end)
        foot_toe  = safe_slice(toe_displacements,  start, end)

        if view == "side":
            hip_rep   = safe_slice(hip_angles,        start, end)
            knee_rep  = safe_slice(knee_angles,        start, end)
            torso_rep = safe_slice(torso_angles,       start, end)
            lean_rep  = safe_slice(torso_lean_angles,  start, end)
            kot_rep   = safe_slice(knee_over_toe,      start, end)
            hbk_rep   = safe_slice(hip_below_knee_arr, start, end)

            if len(hip_rep) < 3 or len(knee_rep) < 3:
                continue

            descent_frames, ascent_frames = compute_descent_ascent(
                signal_for_peaks, idx, tempo_radius
            )
            descent_ascent_ratio = (
                descent_frames / (ascent_frames + 1e-6)
                if ascent_frames > 0 else np.nan
            )

            hip_below_knee_frac = np.nanmean(hbk_rep) if len(hbk_rep) else np.nan

            per_rep_features.append({
                "video_name":          video_name,
                "hip_rom":             np.nanmax(hip_rep)  - np.nanmin(hip_rep),
                "knee_rom":            np.nanmax(knee_rep) - np.nanmin(knee_rep),
                "torso_stability":     np.nanstd(torso_rep),
                "heel_instability":    np.nanstd(foot_heel) if len(foot_heel) else np.nan,
                "toe_instability":     np.nanstd(foot_toe)  if len(foot_toe)  else np.nan,
                "knee_min_angle":      np.nanmin(knee_rep),
                "hip_min_angle":       np.nanmin(hip_rep),
                "torso_lean_peak":     np.nanmax(lean_rep)  if len(lean_rep) else np.nan,
                "torso_lean_mean":     np.nanmean(lean_rep) if len(lean_rep) else np.nan,
                "descent_frames":      float(descent_frames),
                "ascent_frames":       float(ascent_frames),
                "descent_ascent_ratio": descent_ascent_ratio,
                "knee_over_toe_mean":  np.nanmean(kot_rep) if len(kot_rep) else np.nan,
                "knee_over_toe_max":   np.nanmax(kot_rep)  if len(kot_rep) else np.nan,
                "hip_below_knee_frac": hip_below_knee_frac,
            })

        elif view == "front":
            val_rep  = safe_slice(valgus_ratios,        start, end)
            lat_rep  = safe_slice(lateral_lean,         start, end)
            sym_rep  = safe_slice(symmetry,             start, end)
            lat_heel = safe_slice(heel_lateral,         start, end)
            lkx_rep  = safe_slice(left_knee_x_arr,     start, end)
            rkx_rep  = safe_slice(right_knee_x_arr,    start, end)
            lax_rep  = safe_slice(left_ankle_x_arr,    start, end)
            rax_rep  = safe_slice(right_ankle_x_arr,   start, end)
            hip_cx   = safe_slice(hip_center_x_arr,    start, end)
            sho_cx   = safe_slice(shoulder_center_x_arr, start, end)

            if len(val_rep) < 3:
                continue

            VALGUS_CAVE_THRESHOLD = 0.70
            knee_cave_frames = int(np.sum(val_rep < VALGUS_CAVE_THRESHOLD))
            knee_cave_frac   = knee_cave_frames / (len(val_rep) + 1e-6)

            if len(lkx_rep) >= 3 and len(rkx_rep) >= 3:
                stance_w      = np.nanmean(rax_rep - lax_rep) if len(lax_rep) >= 3 else 1.0
                knee_asym     = np.nanmean(np.abs(lkx_rep - rkx_rep) / (stance_w + 1e-6))
                knee_asym_std = np.nanstd(np.abs(lkx_rep  - rkx_rep) / (stance_w + 1e-6))
            else:
                knee_asym = knee_asym_std = np.nan

            if len(lax_rep) >= 3 and len(rax_rep) >= 3:
                stance_widths    = np.abs(rax_rep - lax_rep)
                ankle_width_mean = np.nanmean(stance_widths)
                ankle_width_std  = np.nanstd(stance_widths)
            else:
                ankle_width_mean = ankle_width_std = np.nan

            if len(hip_cx) >= 3:
                hip_shift_max  = np.nanmax(hip_cx) - np.nanmin(hip_cx)
                hip_shift_mean = np.nanstd(hip_cx)
            else:
                hip_shift_max = hip_shift_mean = np.nan

            if len(sho_cx) >= 3 and len(hip_cx) >= 3:
                sho_hip_offset_mean = np.nanmean(np.abs(sho_cx - hip_cx))
                sho_hip_offset_max  = np.nanmax(np.abs(sho_cx - hip_cx))
            else:
                sho_hip_offset_mean = sho_hip_offset_max = np.nan

            per_rep_features.append({
                "video_name":          video_name,
                "valgus_min":          np.nanmin(val_rep),
                "valgus_max":          np.nanmax(val_rep),
                "valgus_variation":    np.nanstd(val_rep),
                "torso_lateral_peak":  np.nanmax(lat_rep)  if len(lat_rep)  else np.nan,
                "symmetry_mean":       np.nanmean(sym_rep) if len(sym_rep)  else np.nan,
                "heel_wobble":         np.nanstd(lat_heel) if len(lat_heel) else np.nan,
                "heel_instability":    np.nanstd(foot_heel) if len(foot_heel) else np.nan,
                "toe_instability":     np.nanstd(foot_toe)  if len(foot_toe)  else np.nan,
                "knee_cave_frames":    float(knee_cave_frames),
                "knee_cave_frac":      knee_cave_frac,
                "knee_asym_mean":      knee_asym,
                "knee_asym_std":       knee_asym_std,
                "ankle_width_mean":    ankle_width_mean,
                "ankle_width_std":     ankle_width_std,
                "hip_shift_max":       hip_shift_max,
                "hip_shift_std":       hip_shift_mean,
                "sho_hip_offset_mean": sho_hip_offset_mean,
                "sho_hip_offset_max":  sho_hip_offset_max,
            })

    return per_rep_features