import os
import cv2

def frames_to_video(frame_folder, output_path, fps=30, resize=None):
    """
    Convert a folder of frames into a video file.

    Args:
        frame_folder (str): Path to folder containing frames.
        output_path (str): Output video file path (.mp4 or .avi).
        fps (int): Frames per second.
        resize (tuple): Optional (width, height) to resize frames.
    """
    frame_files = sorted([
        f for f in os.listdir(frame_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    if not frame_files:
        print(f"No frames found in {frame_folder}")
        return

    # Read first frame to get dimensions
    first_frame = cv2.imread(os.path.join(frame_folder, frame_files[0]))
    if resize:
        first_frame = cv2.resize(first_frame, resize)
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for .mp4
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_file in frame_files:
        frame_path = os.path.join(frame_folder, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Skipping unreadable frame: {frame_file}")
            continue
        if resize:
            frame = cv2.resize(frame, resize)
        video.write(frame)

    video.release()
    print(f"Saved video: {output_path}")

def convert_all_folders_to_videos(root_path, output_root, fps=30, resize=None):
    os.makedirs(output_root, exist_ok=True)

    folders = sorted([
        f for f in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, f))
    ])

    for i, folder in enumerate(folders, start=1):
        folder_path = os.path.join(root_path, folder)
        output_path = os.path.join(output_root, f"{folder}.mp4")
        print(f"[{i}/{len(folders)}] Converting {folder} to video...")
        frames_to_video(folder_path, output_path, fps=fps, resize=resize)

# Paths to your frame datasets
front_frames_root = "dataset/front"
side_frames_root = "dataset/side"

# Paths where videos will be saved
front_videos_output = "videos/front"
side_videos_output = "videos/side"

# Convert front view frames to videos
convert_all_folders_to_videos(front_frames_root, front_videos_output, fps=30, resize=(640, 480))

# Convert side view frames to videos
convert_all_folders_to_videos(side_frames_root, side_videos_output, fps=30, resize=(640, 480))