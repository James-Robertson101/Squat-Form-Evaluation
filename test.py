from extract_frames import extract_frames
from prediction import predict_reps

frame_folder = extract_frames("side_view_squat.mp4", "frames/squat")
results = predict_reps("frames/squat", view="side")

rep1 = results[0]
print(rep1["labels"])