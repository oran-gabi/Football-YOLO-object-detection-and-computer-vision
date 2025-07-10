#speed_and_distance_estimator/speed_and_distance_estimator.py
import numpy as np
import cv2
import sys
sys.path.append('../')
from utils import measure_distance, get_foot_position

class SpeedAndDistanceEstimator:
    def __init__(self, frame_rate=30):
        self.frame_window = 3  # Reduced window for more frequent updates
        self.frame_rate = frame_rate
        self.smoothing_factor = 0.3  # Add smoothing for speed calculations

    def add_speed_and_distance_to_tracks(self, tracks):
        total_distance = {}
        previous_speeds = {}  # For smoothing

        for object, object_tracks in tracks.items():
            if object in ["ball", "referees"]:
                continue

            number_of_frames = len(object_tracks)
            
            # Initialize tracking for all players
            for frame_tracks in object_tracks:
                for track_id in frame_tracks:
                    if object not in total_distance:
                        total_distance[object] = {}
                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0
                    if track_id not in previous_speeds:
                        previous_speeds[track_id] = 0

            # Calculate speeds and distances
            for frame_num in range(0, number_of_frames - 1):
                next_frame = min(frame_num + self.frame_window, number_of_frames - 1)
                
                current_tracks = object_tracks[frame_num]
                next_tracks = object_tracks[next_frame]
                
                for track_id in current_tracks:
                    if track_id not in next_tracks:
                        continue

                    start_position = current_tracks[track_id].get('position_transformed')
                    end_position = next_tracks[track_id].get('position_transformed')

                    if start_position is None or end_position is None:
                        continue
                    
                    # Calculate distance and speed
                    distance_covered = measure_distance(start_position, end_position)
                    time_elapsed = (next_frame - frame_num) / self.frame_rate
                    
                    if time_elapsed > 0:
                        speed_mps = distance_covered / time_elapsed
                        speed_kmh = speed_mps * 3.6
                        
                        # Apply smoothing
                        smoothed_speed = (speed_kmh * self.smoothing_factor + 
                                        previous_speeds[track_id] * (1 - self.smoothing_factor))
                        previous_speeds[track_id] = smoothed_speed
                    else:
                        smoothed_speed = previous_speeds[track_id]

                    total_distance[object][track_id] += distance_covered

                    # Update all frames in the window
                    for frame_idx in range(frame_num, next_frame + 1):
                        if (frame_idx < len(object_tracks) and 
                            track_id in object_tracks[frame_idx]):
                            object_tracks[frame_idx][track_id].update({
                                'speed': smoothed_speed,
                                'distance': total_distance[object][track_id]
                            })

    def draw_speed_and_distance(self, frames, tracks):
        output_frames = []
        
        for frame_num, frame in enumerate(frames):
            frame_copy = frame.copy()
            
            for object, object_tracks in tracks.items():
                if object in ["ball", "referees"]:
                    continue

                if frame_num < len(object_tracks):
                    for track_id, track_info in object_tracks[frame_num].items():
                        speed = track_info.get('speed')
                        distance = track_info.get('distance')
                        bbox = track_info.get('bbox')
                        
                        if speed is None or distance is None or bbox is None:
                            continue
                        
                        # Improved text placement
                        foot_pos = get_foot_position(bbox)
                        if foot_pos is None:
                            continue
                        
                        # Draw with better visibility
                        x, y = int(foot_pos[0]), int(foot_pos[1])
                        
                        # Draw background rectangle for better visibility
                        text_bg = np.zeros((40, 100, 3), dtype=np.uint8)
                        cv2.putText(text_bg,
                                  f"{speed:.1f} km/h",
                                  (5, 15),
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  0.5,
                                  (255, 255, 255),
                                  1)
                        cv2.putText(text_bg,
                                  f"{distance:.1f} m",
                                  (5, 35),
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  0.5,
                                  (255, 255, 255),
                                  1)
                        
                        # Blend text background with frame
                        y_offset = y + 10
                        x_offset = x - 50
                        
                        # Ensure coordinates are within frame boundaries
                        if (0 <= y_offset < frame_copy.shape[0] - 40 and 
                            0 <= x_offset < frame_copy.shape[1] - 100):
                            roi = frame_copy[y_offset:y_offset + 40, 
                                          x_offset:x_offset + 100]
                            frame_copy[y_offset:y_offset + 40,
                                     x_offset:x_offset + 100] = \
                                cv2.addWeighted(roi, 0.5, text_bg, 0.5, 0)
            
            output_frames.append(frame_copy)

        return output_frames



# import cv2
# import sys
# import numpy as np
# sys.path.append('../')
# from utils import measure_distance, get_foot_position

# class SpeedAndDistance_Estimator():
#     def __init__(self):
#         self.frame_window = 3
#         self.frame_rate = 24
#         self.pixel_to_meter = 0.015
#         self.speed_smoothing_factor = 0.8
#         self.metrics_memory = {}  # Store metrics between frames
#         self.max_pixel_movement = 100  # Filter threshold for distance in pixels

#     def add_speed_and_distance_to_tracks(self, tracks):
#         total_distance = {}
#         prev_speeds = {}

#         for object, object_tracks in tracks.items():
#             if object in ["ball", "referees"]:
#                 continue

#             number_of_frames = len(object_tracks)
            
#             for frame_num in range(number_of_frames):
#                 for track_id, track_info in object_tracks[frame_num].items():
#                     track_key = (object, track_id)
#                     self._initialize_metrics(track_key, total_distance, track_info)

#                     # Skip invalid bounding boxes
#                     current_bbox = track_info.get('bbox')
#                     if not current_bbox:
#                         self._use_last_known_metrics(frame_num, track_key, tracks, object, track_id)
#                         continue

#                     next_track = self._find_next_valid_track(object_tracks, track_id, frame_num, number_of_frames)
#                     if not next_track:
#                         self._use_last_known_metrics(frame_num, track_key, tracks, object, track_id)
#                         continue

#                     # Calculate and validate metrics
#                     speed_km_per_hour, distance_meters = self._calculate_metrics(current_bbox, next_track['bbox'], track_key, prev_speeds)

#                     if speed_km_per_hour is not None:
#                         self._update_metrics(frame_num, object, track_id, speed_km_per_hour, distance_meters, tracks, total_distance)

#         return tracks

#     def _initialize_metrics(self, track_key, total_distance, track_info):
#         if track_key not in self.metrics_memory:
#             self.metrics_memory[track_key] = {'speed': 0, 'distance': 0}
#             total_distance.setdefault(track_key[0], {})[track_key[1]] = 0

#     def _use_last_known_metrics(self, frame_num, track_key, tracks, object, track_id):
#         if track_key in self.metrics_memory:
#             metrics = self.metrics_memory[track_key]
#             tracks[object][frame_num][track_id]['speed'] = metrics['speed']
#             tracks[object][frame_num][track_id]['distance'] = metrics['distance']

#     def _find_next_valid_track(self, object_tracks, track_id, frame_num, number_of_frames):
#         frames_checked = 0
#         for next_frame in range(frame_num + 1, min(number_of_frames, frame_num + self.frame_window + 1)):
#             if track_id in object_tracks[next_frame] and 'bbox' in object_tracks[next_frame][track_id]:
#                 return object_tracks[next_frame][track_id]
#             frames_checked += 1
#         return None

#     def _calculate_metrics(self, current_bbox, next_bbox, track_key, prev_speeds):
#         dx, dy = (next_bbox[0] - current_bbox[0], next_bbox[1] - current_bbox[1])
#         distance_pixels = np.hypot(dx, dy)

#         if distance_pixels > self.max_pixel_movement:
#             return None, None

#         distance_meters = distance_pixels * self.pixel_to_meter
#         speed_mps = distance_meters * self.frame_rate / self.frame_window
#         speed_kmph = min(max(0, speed_mps * 3.6), 35)

#         if track_key in prev_speeds:
#             speed_kmph = (prev_speeds[track_key] * self.speed_smoothing_factor +
#                           speed_kmph * (1 - self.speed_smoothing_factor))

#         prev_speeds[track_key] = speed_kmph
#         return speed_kmph, distance_meters

#     def _update_metrics(self, frame_num, object, track_id, speed, distance, tracks, total_distance):
#         track_key = (object, track_id)
#         total_distance[object][track_id] += distance
#         tracks[object][frame_num][track_id]['speed'] = speed
#         tracks[object][frame_num][track_id]['distance'] = total_distance[object][track_id]
#         self.metrics_memory[track_key] = {'speed': speed, 'distance': total_distance[object][track_id]}

#     def draw_speed_and_distance(self, frames, tracks):
#         font, font_scale, thickness, padding = cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1, 2
#         for frame_num, frame in enumerate(frames):
#             frame_copy = frame.copy()
#             for object, object_tracks in tracks.items():
#                 if object in ["ball", "referees"] or frame_num >= len(object_tracks):
#                     continue
#                 for track_id, track_info in object_tracks[frame_num].items():
#                     bbox, speed, distance = (track_info.get(k) for k in ['bbox', 'speed', 'distance'])
#                     if bbox and speed is not None and distance is not None:
#                         self._draw_metrics(frame_copy, bbox, int(speed), int(distance), font, font_scale, thickness, padding)
#             frames[frame_num] = frame_copy
#         return frames

#     def _draw_metrics(self, frame, bbox, speed, distance, font, font_scale, thickness, padding):
#         x1, y1, x2, y2 = map(int, bbox)
#         base_y = y2 + 20
#         for text, offset_x in [(f"{speed} km/h", -15), (f"{distance} m", 15)]:
#             text_w, text_h = cv2.getTextSize(text, font, font_scale, thickness)[0]
#             pos = ((x1 + x2) // 2 + offset_x - text_w // 2, base_y + text_h)
#             cv2.putText(frame, text, pos, font, font_scale, (0, 0, 0), thickness, lineType=cv2.LINE_AA)



