import cv2
import os, shutil

def extract_frames(video_path, output_folder, fps=30, overwrite = True):
    """
    Splits a video into individual frames saved as JPGs.
    
    fps: how many frames per second to extract.
    """
    if overwrite and os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    video_fps    = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, round(video_fps / fps))  # skip frames to hit target fps

    frame_count  = 0
    saved_count  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            filename = os.path.join(output_folder, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {saved_count} frames to {output_folder}")
    return output_folder