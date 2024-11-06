# video_utils.py

import cv2
import os


# creat a object in the vidoe using cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Get the frame rate from the input video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()  # Release the video capture object
    return frames, frame_rate  # Return both frames and frame rate

def save_video(output_video_frames, output_video_path, frame_rate):
    # Check if the output file already exists and remove it if it does
    if os.path.exists(output_video_path):
        os.remove(output_video_path)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    height, width = output_video_frames[0].shape[:2]
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))
    
    for frame in output_video_frames:
        out.write(frame)
    
    out.release()  # Release the video writer object