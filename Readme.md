source:
https://www.youtube.com/watch?v=neBZ6huolkg    1:37:00

https://colab.research.google.com/drive/1tiesfrBUp1yy-lneIKp1yM65XqOSmEja#scrollTo=KW8FT73g9IdO







Training :
To pause and resume training in YOLOv5 on Google Colab, you can use checkpoint files. YOLOv5 automatically saves checkpoint weights after each epoch if you specify a save directory. When resuming, it picks up from the last checkpoint, so there’s no need to start over.
Here’s how to set up your training code to support pausing and resuming:
1. Setup for Initial Training (Save Checkpoints)
In your training command, specify the following options:
--save-period 1: Save weights every epoch (checkpoint files will be created).
--project <drive checkpoint path>: Save these checkpoints to a path on your Google Drive so they persist after Colab disconnects.
Example:
python
Copy code
!python train.py --img 640 --batch 16 --epochs 100 --data "/content/football-players-detection-1/data.yaml" \
                 --weights "yolov5s.pt" --project "/content/drive/MyDrive/Football_YOLO_Checkpoints" \
                 --name "football_yolo_training" --save-period 1








2. Checkpoint Files Saved on Google Drive
This command will save weights as:
Football_YOLO_Checkpoints/football_yolo_training/weights/last.pt (the latest weights)
Football_YOLO_Checkpoints/football_yolo_training/weights/best.pt (best weights based on performance)


3. Resume Training from Last Checkpoint
To resume from the last saved checkpoint, specify the path to last.pt in the --weights argument:
python
Copy code
!python train.py --img 640 --batch 16 --epochs 100 --data "/content/football-players-detection-1/data.yaml" \
                 --weights "/content/drive/MyDrive/Football_YOLO_Checkpoints/football_yolo_training/weights/last.pt" \
                 --project "/content/drive/MyDrive/Football_YOLO_Checkpoints" --name "football_yolo_training" \
                 --save-period 1

4. Explanation of Commands
--weights "/content/drive/MyDrive/Football_YOLO_Checkpoints/football_yolo_training/weights/last.pt": This points to the latest checkpoint, allowing YOLOv5 to resume from where it left off.
--epochs 100: YOLOv5 will continue training until it reaches 100 epochs, counting from the saved checkpoint.
How to Pause Training
Simply stop the training cell in Colab whenever you wish to pause.
Reconnect to Google Drive in Colab and run the resume command to continue training from the last saved checkpoint.
This way, you can start, pause, and resume training on Colab using Google Drive to store your checkpoints!
Creating the output_video for the objects trackers  :

# main.py
import os
from utils import read_video, save_video
from trackers import Tracker


def main():
    # Define paths
    input_path = 'input_videos/08fd33_4.mp4'
    output_dir = 'output_videos'
    output_path = f'{output_dir}/output_video.avi'
   
    # Check if input video exists
    if not os.path.exists(input_path):
        print(f"Error: Input video '{input_path}' does not exist.")
        return


    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
   
    # Read Video
    video_frames = read_video(input_path)
    if not video_frames:
        print("Error: No frames read from the input video.")
        return


    # Save video
    save_video(video_frames, output_path)
    print(f"Video saved successfully to {output_path}")


if __name__ == '__main__':
    main()








Creating the tracker and object id with dictionary list of new objects:
The Tracker class provided is designed to perform object detection and tracking on frames using the YOLO model from the ultralytics library, and to organize the detections by object classes like "players," "referees," and "ball." Here's a summary and explanation of each component, including notes for understanding how the code functions:
Summary of Key Functionalities
Initialization (__init__ method):
Initializes a YOLO model using a specified model_path.
Sets up a tracker using ByteTrack from the supervision library (assumed as a tracking algorithm).
Frame Detection (detect_frames method):
Processes a batch of frames to detect objects, using a batch_size of 20 for efficiency.
Calls self.model.predict() to run object detection on each batch and returns a list of detections.
Object Tracking (get_object_tracks method):
Handles object tracking over a series of frames.
Supports reading pre-computed tracks from a file (stub) to save time on re-computation.
Assigns detections to specific classes ("players," "referees," "ball"), using their tracking IDs to keep consistent tracking across frames.

from ultralytics import YOLO
import supervision as sv
import pickle
import os


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()  # Assuming ByteTrack from `supervision` is being used


    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i + batch_size], conf=0.1)
            # detections.extend(detections_batch)  # Append to detections list
            detections += detections_batch
        return detections


    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):


        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks =pickle.load(f)
                return tracks


        detections = self.detect_frames(frames)


        tracks={
            "players":[],
            "referees":[],
            "ball":[]


        }


        for frame_num, detection in enumerate(detections):
            cls_names = detection.names  # Class names dictionary
            cls_names_inv = {v: k for k, v in cls_names.items()}  # Invert class names dictionary
            print(cls_names)


            # Convert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)




            # Convert Goalkeeper to player object
            for object_ind , cls_id in enumerate(detection_supervision.cls_id):
                if cls_names[cls_id] == "goalkeeper":
                    detection_supervision.cls_id[object_ind] = cls_names_inv['player']






            # Track Object
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)


            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})


            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]


                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}


                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}


            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]


                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}


        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)




        return tracks    



