import cv2
import numpy as np
import mediapipe as mp
from modules.calculateAngle import calculate_angle
from modules.midpoint import midpoint
from modules.smoothAngle import smooth_angle
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
import os

def extract_squat_features(video_path, min_rom=20, view="side", show_video=False, draw_keypoints=False):
    """
    Extract per-rep squat features from a video (side or front view).

    Args:
        video_path (str): Path to the video file.
        min_rom (float): Minimum hip ROM to count as a valid rep.
        view (str): "side" or "front"
        show_video (bool): Whether to display video with overlays.
        draw_keypoints (bool): Whether to draw mediapipe keypoints.

    Returns:
        List[dict]: Per-rep features with video_name included.
    """
    video_name = os.path.basename(video_path)
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    pose = mp_pose.Pose(static_image_mode=False,
                        model_complexity=1,
                        enable_segmentation=False,
                        min_detection_confidence=0.1,
                        min_tracking_confidence=0.1)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    prev_knee = prev_hip = prev_torso = None

    hip_angles = []
    knee_angles = []
    torso_angles = []
    lateral_lean = []  # front view
    symmetry = []      # front view
    hip_y_positions = []  # for rep detection in front view
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (640, 480))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame_resized.shape

            # Keypoints
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h]
            hip = midpoint(left_hip, right_hip)

            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
            shoulder = midpoint(left_shoulder, right_shoulder)

            ear_landmark = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
            left_shoulder_vis = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility
            right_shoulder_vis = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility

            if ear_landmark.visibility >= 0.5:
                torso_top = [ear_landmark.x * w, ear_landmark.y * h]
            elif left_shoulder_vis >= 0.5 or right_shoulder_vis >= 0.5:
                torso_top = shoulder
            else:
                vector = np.array(shoulder) - np.array(hip)
                torso_top = (np.array(shoulder) + vector * 0.5).tolist()

            # Side view: sagittal angles
            if view == "side":
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * w,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * h]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h]

                knee_angle = calculate_angle(hip, knee, ankle)
                hip_flexion_angle = calculate_angle(shoulder, hip, knee)
                torso_angle = calculate_angle(hip, shoulder, torso_top)

                smoothed_knee = smooth_angle(knee_angle, prev_knee)
                smoothed_hip = smooth_angle(hip_flexion_angle, prev_hip)
                smoothed_torso = smooth_angle(torso_angle, prev_torso)

                prev_knee, prev_hip, prev_torso = smoothed_knee, smoothed_hip, smoothed_torso

                hip_angles.append(smoothed_hip)
                knee_angles.append(smoothed_knee)
                torso_angles.append(smoothed_torso)

            # Front view: lateral features
            elif view == "front":
                hip_y_positions.append(hip[1])
                knee_angle_front = calculate_angle([hip[0], hip[1]],
                                                   [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * w,
                                                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * h],
                                                   [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w,
                                                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * h])
                hip_angles.append(knee_angle_front)

                torso_mid = midpoint(left_shoulder, right_shoulder)
                hip_mid = midpoint(left_hip, right_hip)
                lateral_lean.append(abs(torso_mid[0] - hip_mid[0]))

                shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
                hip_width = abs(left_hip[0] - right_hip[0])
                symmetry.append(shoulder_width / hip_width)

            if show_video:
                overlay = frame_resized.copy()
                if draw_keypoints:
                    mp_drawing.draw_landmarks(
                        overlay,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
                    )
                cv2.imshow("Pose Detection", overlay)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()

    # Post-processing: rep detection
    hip_array = np.array(hip_y_positions if view == "front" else hip_angles)
    frame_times = np.arange(len(hip_array)) / fps
    hip_smooth = savgol_filter(hip_array, window_length=15, polyorder=2)

    min_indices, _ = find_peaks(-hip_smooth, distance=int(fps*0.5))
    max_indices, _ = find_peaks(hip_smooth, distance=int(fps*0.5))

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

        if view == "side":
            hip_rep = np.array(hip_angles[start:end])
            knee_rep = np.array(knee_angles[start:end])
            torso_rep = np.array(torso_angles[start:end])
            time_rep = frame_times[start:end]

            hip_speed = np.max(np.abs(np.diff(hip_rep)) / np.diff(time_rep))
            knee_speed = np.max(np.abs(np.diff(knee_rep)) / np.diff(time_rep))
            torso_peak = np.min(torso_rep)

            per_rep_features.append({
                'video_name': video_name,
                'hip_min': np.min(hip_rep),
                'hip_max': np.max(hip_rep),
                'hip_rom': hip_rom,
                'hip_speed': hip_speed,
                'knee_min': np.min(knee_rep),
                'knee_max': np.max(knee_rep),
                'knee_rom': np.max(knee_rep) - np.min(knee_rep),
                'knee_speed': knee_speed,
                'torso_peak': torso_peak
            })
        elif view == "front":
            lateral_rep = np.array(lateral_lean[start:end])
            symmetry_rep = np.array(symmetry[start:end])
            knee_rep = np.array(hip_angles[start:end])

            per_rep_features.append({
                'video_name': video_name,
                'knee_lateral_min': np.min(knee_rep),
                'knee_lateral_max': np.max(knee_rep),
                'knee_lateral_rom': np.max(knee_rep) - np.min(knee_rep),
                'torso_lateral_peak': np.max(lateral_rep),
                'symmetry_mean': np.mean(symmetry_rep)
            })

    return per_rep_features
