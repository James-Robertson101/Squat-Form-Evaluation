import cv2
import numpy as np
import os
from scipy.signal import find_peaks, savgol_filter
from modules.calculateAngle import calculate_angle
from modules.midpoint import midpoint
from modules.smoothAngle import smooth_angle

import mediapipe as mp
mp_pose_module = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_pose_class = mp_pose_module.Pose

# Cache landmark indices once
LH, RH = mp_pose_module.PoseLandmark.LEFT_HIP.value, mp_pose_module.PoseLandmark.RIGHT_HIP.value
LS, RS = mp_pose_module.PoseLandmark.LEFT_SHOULDER.value, mp_pose_module.PoseLandmark.RIGHT_SHOULDER.value
LK, LA = mp_pose_module.PoseLandmark.LEFT_KNEE.value, mp_pose_module.PoseLandmark.LEFT_ANKLE.value
LE = mp_pose_module.PoseLandmark.LEFT_EAR.value

def extract_squat_features_from_frames(frame_folder, min_rom=20, view="side", draw_keypoints=False):
    """
    Extract per-rep squat features from a folder of frames (side or front view).
    """

    video_name = os.path.basename(frame_folder.rstrip("/\\"))
    pose = mp_pose_class(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.1,
        min_tracking_confidence=0.1
    )

    frame_files = sorted([f for f in os.listdir(frame_folder)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    fps = 30
    prev_knee = prev_hip = prev_torso = None

    hip_angles, knee_angles, torso_angles = [], [], []
    lateral_lean, symmetry, hip_y_positions = [], [], []
    valgus_ratios = []

    for frame_file in frame_files:
        frame_path = os.path.join(frame_folder, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        frame_resized = cv2.resize(frame, (640, 480))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if not results.pose_landmarks:
            continue

        landmarks = results.pose_landmarks.landmark
        h, w, _ = frame_resized.shape

        # ---- Keypoints ----
        left_hip = np.array([landmarks[LH].x * w, landmarks[LH].y * h])
        right_hip = np.array([landmarks[RH].x * w, landmarks[RH].y * h])
        hip = midpoint(left_hip, right_hip)

        left_shoulder = np.array([landmarks[LS].x * w, landmarks[LS].y * h])
        right_shoulder = np.array([landmarks[RS].x * w, landmarks[RS].y * h])
        shoulder = midpoint(left_shoulder, right_shoulder)

        ear_landmark = landmarks[LE]
        left_shoulder_vis = landmarks[LS].visibility
        right_shoulder_vis = landmarks[RS].visibility

        if ear_landmark.visibility >= 0.5:
            torso_top = np.array([ear_landmark.x * w, ear_landmark.y * h])
        elif left_shoulder_vis >= 0.5 or right_shoulder_vis >= 0.5:
            torso_top = shoulder
        else:
            torso_top = shoulder + (shoulder - hip) * 0.5

        # =========================
        # SIDE VIEW FEATURES
        # =========================
        if view == "side":

            knee = np.array([landmarks[LK].x * w, landmarks[LK].y * h])
            ankle = np.array([landmarks[LA].x * w, landmarks[LA].y * h])

            knee_angle = calculate_angle(hip, knee, ankle)
            hip_angle = calculate_angle(shoulder, hip, knee)
            torso_angle = calculate_angle(hip, shoulder, torso_top)

            smoothed_knee = smooth_angle(knee_angle, prev_knee)
            smoothed_hip = smooth_angle(hip_angle, prev_hip)
            smoothed_torso = smooth_angle(torso_angle, prev_torso)

            prev_knee, prev_hip, prev_torso = smoothed_knee, smoothed_hip, smoothed_torso

            hip_angles.append(smoothed_hip)
            knee_angles.append(smoothed_knee)
            torso_angles.append(smoothed_torso)

        # =========================
        # FRONT VIEW FEATURES
        # =========================
        elif view == "front":

            hip_y_positions.append(hip[1])

            left_knee = np.array([landmarks[LK].x * w, landmarks[LK].y * h])
            right_knee = np.array([landmarks[mp_pose_module.PoseLandmark.RIGHT_KNEE.value].x * w,
                                   landmarks[mp_pose_module.PoseLandmark.RIGHT_KNEE.value].y * h])

            knee_width = abs(left_knee[0] - right_knee[0])
            hip_width = abs(left_hip[0] - right_hip[0])
            valgus_ratio = knee_width / (hip_width + 1e-6)
            valgus_ratios.append(valgus_ratio)

            torso_mid = midpoint(left_shoulder, right_shoulder)
            hip_mid = midpoint(left_hip, right_hip)
            lateral_lean.append(abs(torso_mid[0] - hip_mid[0]))

            shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
            symmetry.append(shoulder_width / (hip_width + 1e-6))

        if draw_keypoints:
            overlay = frame_resized.copy()
            mp_drawing.draw_landmarks(
                overlay,
                results.pose_landmarks,
                mp_pose_module.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            cv2.imshow("Pose Detection", overlay)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    pose.close()
    cv2.destroyAllWindows()

    # Convert to arrays
    hip_angles = np.array(hip_angles)
    knee_angles = np.array(knee_angles)
    torso_angles = np.array(torso_angles)
    lateral_lean = np.array(lateral_lean)
    symmetry = np.array(symmetry)
    hip_y_positions = np.array(hip_y_positions)
    valgus_ratios = np.array(valgus_ratios)

    hip_array = hip_y_positions if view == "front" else hip_angles
    frame_times = np.arange(len(hip_array)) / fps

    if len(hip_array) < 5:
        return []

    window_len = min(15, len(hip_array) // 2 * 2 + 1)
    hip_smooth = savgol_filter(hip_array, window_length=window_len, polyorder=2)

    min_indices, _ = find_peaks(-hip_smooth, distance=int(fps * 0.5))
    max_indices, _ = find_peaks(hip_smooth, distance=int(fps * 0.5))

    per_rep_features = []

    for bottom_idx in min_indices:
        prev_max_candidates = max_indices[max_indices < bottom_idx]
        if len(prev_max_candidates) == 0:
            continue

        top_idx = prev_max_candidates[-1]
        hip_min = hip_array[bottom_idx]
        hip_max = hip_array[top_idx]
        hip_rom = hip_max - hip_min

        if hip_rom < min_rom:
            continue

        start, end = top_idx, bottom_idx + 1

        # =========================
        # SIDE REP FEATURES
        # =========================
        if view == "side":

            hip_rep = hip_angles[start:end]
            knee_rep = knee_angles[start:end]
            torso_rep = torso_angles[start:end]
            time_rep = frame_times[start:end]

            bottom_local = np.argmin(hip_rep)

            descent_time = time_rep[bottom_local] - time_rep[0]
            ascent_time = time_rep[-1] - time_rep[bottom_local]
            rep_duration = time_rep[-1] - time_rep[0]

            hip_vel = np.diff(hip_rep) / np.diff(time_rep)
            knee_vel = np.diff(knee_rep) / np.diff(time_rep)

            hip_peak_velocity = np.max(np.abs(hip_vel))
            knee_peak_velocity = np.max(np.abs(knee_vel))

            hip_acc = np.diff(hip_vel) / np.diff(time_rep[:-1])
            hip_peak_acceleration = np.max(np.abs(hip_acc)) if len(hip_acc) > 0 else 0

            torso_stability = np.std(torso_rep)
            hip_stability = np.std(hip_rep)

            torso_angle_bottom = torso_rep[bottom_local]
            knee_rom = np.max(knee_rep) - np.min(knee_rep)
            hip_knee_rom_ratio = hip_rom / (knee_rom + 1e-6)

            per_rep_features.append({
                'video_name': video_name,
                'hip_min': np.min(hip_rep),
                'hip_max': np.max(hip_rep),
                'hip_rom': hip_rom,
                'knee_min': np.min(knee_rep),
                'knee_max': np.max(knee_rep),
                'knee_rom': knee_rom,
                'torso_peak': np.min(torso_rep),

                'descent_time': descent_time,
                'ascent_time': ascent_time,
                'rep_duration': rep_duration,

                'hip_peak_velocity': hip_peak_velocity,
                'knee_peak_velocity': knee_peak_velocity,
                'hip_peak_acceleration': hip_peak_acceleration,

                'torso_stability': torso_stability,
                'hip_stability': hip_stability,

                'torso_angle_bottom': torso_angle_bottom,
                'hip_knee_rom_ratio': hip_knee_rom_ratio
            })

        # =========================
        # FRONT REP FEATURES
        # =========================
        elif view == "front":

            lateral_rep = lateral_lean[start:end]
            symmetry_rep = symmetry[start:end]
            valgus_rep = valgus_ratios[start:end]

            per_rep_features.append({
                'video_name': video_name,
                'valgus_min': np.min(valgus_rep),
                'valgus_max': np.max(valgus_rep),
                'valgus_variation': np.std(valgus_rep),

                'torso_lateral_peak': np.max(lateral_rep),
                'symmetry_mean': np.mean(symmetry_rep)
            })

    return per_rep_features
