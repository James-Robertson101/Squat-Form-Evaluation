import cv2
import os
import shutil


def extract_frames(video_path, output_folder, fps=30, overwrite=True):
    """
    Splits a video into individual frames saved as JPGs.
    Automatically fixes rotated phone videos when possible.
    """
    if overwrite and os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, round(video_fps / fps))

    # Try to detect rotation metadata
    try:
        rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
    except:
        rotation = 0

    print(f"Detected rotation: {rotation}")

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply rotation correction if needed
        if rotation == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        elif rotation == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        elif rotation == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if frame_count % frame_interval == 0:
            filename = os.path.join(
                output_folder,
                f"frame_{saved_count:05d}.jpg"
            )
            cv2.imwrite(filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {saved_count} frames to {output_folder}")
    return output_folder