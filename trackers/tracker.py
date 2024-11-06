from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import cv2
import sys
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_bbox_height

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.referee_track_ids = set()
        self.ball_track_ids = set()  # Track IDs specifically for the ball

    def is_ball(self, bbox, cls_id, cls_names_inv, frame):
        if cls_id != cls_names_inv['ball']:  # Assuming 'ball' is the correct label for your ball
            print(f"Detected object is not a ball, cls_id: {cls_id}")
            return False

        width = get_bbox_width(bbox)
        height = get_bbox_height(bbox)

        # Set the maximum size for a bounding box to be considered as a ball
        max_ball_size = 50  
        min_aspect_ratio = 0.75  
        max_aspect_ratio = 1.3   
        
        aspect_ratio = width / height
        if (
            width < max_ball_size 
            and height < max_ball_size 
            and min_aspect_ratio <= aspect_ratio <= max_aspect_ratio
         ):
            if self.is_ball_color(frame, bbox):
              print(f"Ball detected with bbox {bbox}")  
              return True
            else:
             print(f"Ball color check failed for bbox {bbox}")
        else:
             print(f"Ball size/aspect ratio check failed for bbox {bbox}")

        return False

    def is_ball_color(self, frame, bbox):
        # Extract the region of interest (ROI) based on bbox
        x1, y1, x2, y2 = map(int, bbox)
        roi = frame[y1:y2, x1:x2]

        # Convert ROI to HSV color space for color detection
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Define HSV range for typical ball colors
        lower_white = np.array([0, 0, 180])  # Adjust based on your ball color
        upper_white = np.array([180, 30, 255])
        
        # Create mask for white color
        mask_white = cv2.inRange(hsv_roi, lower_white, upper_white)
        
        # Calculate the percentage of white pixels in ROI
        white_ratio = cv2.countNonZero(mask_white) / (roi.size / 3)  # Normalize by ROI size
        
        # If a significant portion of ROI is white, it may be the ball
        return white_ratio > 0.5

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i + batch_size], conf=0.3)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
                return tracks

        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}
            
            detection_supervision = sv.Detections.from_ultralytics(detection)

            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv['player']

            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if track_id in self.referee_track_ids:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}
                    continue

                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}
                    self.referee_track_ids.add(track_id)
                elif cls_id == cls_names_inv['player'] and track_id not in self.referee_track_ids:
                    if self.is_ball(bbox, cls_id, cls_names_inv, frames[frame_num]):
                        # Reclassify as ball based on color and size checks
                        tracks["ball"][frame_num][track_id] = {"bbox": bbox}
                        self.ball_track_ids.add(track_id)
                        print(f"Ball detected in frame {frame_num} with bbox {bbox}")
                    else:
                        tracks["players"][frame_num][track_id] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        if track_id is not None:
            rectangle_width = 40
            rectangle_height = 20
            x1_rect = x_center - rectangle_width // 2
            x2_rect = x_center + rectangle_width // 2
            y1_rect = (y2 - rectangle_height // 2) + 15
            y2_rect = (y2 + rectangle_height // 2) + 15

            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y + 20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

def draw_annotations(self, video_frames, tracks):
    output_video_frames = []

    for frame_num, frame in enumerate(video_frames):
        frame = frame.copy()
        
        player_dict = tracks["players"][frame_num] if frame_num < len(tracks["players"]) else {}
        ball_dict = tracks["ball"][frame_num] if frame_num < len(tracks["ball"]) else {}
        referee_dict = tracks["referees"][frame_num] if frame_num < len(tracks["referees"]) else {}

        print(f"Frame {frame_num}: Players={len(player_dict)}, Ball={len(ball_dict)}, Referees={len(referee_dict)}")

        # Draw referees using the same ellipse shape as players but in yellow and without numbers
        for track_id, referee in referee_dict.items():
            self.draw_referee(frame, referee['bbox'], track_id)

        # Draw players
        for track_id, player in player_dict.items():
            if track_id not in self.referee_track_ids:
                self.draw_ellipse(frame, player['bbox'], (0, 0, 255), track_id)

        # Draw ball last (top layer)
        for _, ball in ball_dict.items():
            self.draw_ellipse(frame, ball['bbox'], (255, 0, 0))

        # Check if ball is detected, if not, draw a triangle marker
        if len(ball_dict) == 0:
            # Assume the ball is at the center of the field
            center_x = frame.shape[1] // 2
            center_y = frame.shape[0] // 2
            
            # Draw a triangle marker around the assumed ball position
            pts = np.array([[center_x-20, center_y+20], [center_x, center_y-20], [center_x+20, center_y+20]], np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

        output_video_frames.append(frame)

    print(f"Output frames generated: {len(output_video_frames)}, Expected: {len(video_frames)}")
    return output_video_frames
