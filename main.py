# main.py
import os
import numpy as np  # Make sure to import numpy if you're using it
from utils import read_video, save_video
from trackers import Tracker

def main():
    try:
        model_path = 'models/best.pt'
        tracker = Tracker(model_path)

        video_frames, frame_rate = read_video('input_videos/08fd33_4.mp4')
        print("Read video frames:", len(video_frames), "at frame rate:", frame_rate)

        tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')
        output_video_frames = tracker.draw_annotations(video_frames, tracks)

        print("Output video frames count:", len(output_video_frames))
        
        output_path = 'output_videos/output_video.avi'
        save_video(output_video_frames, output_path, frame_rate)
        print(f"Video saved to {output_path}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()