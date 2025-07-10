# main.py
from utils import read_video, save_video
from trackers import Tracker
import os
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from utils.bbox_utils import measure_distance
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistanceEstimator  # Changed to match class name

def main():
    input_video_path = 'input_videos/08fd33_4.mp4'
    
    # Get frame rate from input video
    input_video = cv2.VideoCapture(input_video_path)
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    total_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
    input_video.release()
    
    # Read Video - limit to actual frame count
    video_frames = read_video(input_video_path)[:total_frames]

    # Initialize Tracker
    tracker = Tracker('models/best.pt')

    # Get tracks with improved detection parameters
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/track_stubs.pkl'
    )
    
    # Initialize player IDs for both teams (22 players total)
    all_player_ids = set()
    for frame_tracks in tracks['players']:
        all_player_ids.update(frame_tracks.keys())
    
    # Ensure consistent player tracking
    for frame_num in range(len(tracks['players'])):
        for player_id in all_player_ids:
            if player_id not in tracks['players'][frame_num]:
                # If player not detected, interpolate from nearby frames
                prev_frame = next((i for i in range(frame_num-1, -1, -1) 
                                   if player_id in tracks['players'][i]), None)
                next_frame = next((i for i in range(frame_num+1, len(tracks['players'])) 
                                   if player_id in tracks['players'][i]), None)
                
                if prev_frame is not None and next_frame is not None:
                    # Linear interpolation of position
                    prev_pos = tracks['players'][prev_frame][player_id]['bbox']
                    next_pos = tracks['players'][next_frame][player_id]['bbox']
                    alpha = (frame_num - prev_frame) / (next_frame - prev_frame)
                    
                    interpolated_bbox = [
                        prev_pos[i] + alpha * (next_pos[i] - prev_pos[i])
                        for i in range(4)
                    ]
                    
                    tracks['players'][frame_num][player_id] = {
                        'bbox': interpolated_bbox,
                        'conf': 0.5  # Lower confidence for interpolated positions
                    }

    # Get object positions
    tracker.add_position_to_tracks(tracks)

    # Camera movement estimation
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/camera_movement_stub.pkl'
    )
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # View Transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(video_frames)
    

    # Speed and distance estimation
    speed_and_distance_estimator = SpeedAndDistanceEstimator(frame_rate=fps)
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Team Assignment
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    
    # Assign teams to all players consistently
    player_teams = {}  # Store consistent team assignments
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            if player_id not in player_teams:
                team = team_assigner.get_player_team(
                    video_frames[frame_num],
                    track['bbox'],
                    player_id
                )
                player_teams[player_id] = team
            
            tracks['players'][frame_num][player_id]['team'] = player_teams[player_id]
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[player_teams[player_id]]
    
    # Ball possession tracking
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    default_team = None
    possession_threshold = 0.3  # meters
    
    for frame_num in range(len(video_frames)):
        if frame_num < len(tracks['ball']) and 1 in tracks['ball'][frame_num]:
            ball_pos = tracks['ball'][frame_num][1].get('position_transformed')
            if ball_pos is not None:
                closest_player = None
                min_distance = float('inf')
                for player_id, track in tracks['players'][frame_num].items():
                    player_pos = track.get('position_transformed')
                    if player_pos is not None:
                        distance = measure_distance(ball_pos, player_pos)
                        if distance < min_distance:
                            min_distance = distance
                            closest_player = player_id
                
                if closest_player is not None and min_distance < possession_threshold:
                    current_team = tracks['players'][frame_num][closest_player]['team']
                    tracks['players'][frame_num][closest_player]['has_ball'] = True
                    team_ball_control.append(current_team)
                    default_team = current_team
                else:
                    team_ball_control.append(default_team if default_team is not None else 1)
            else:
                team_ball_control.append(default_team if default_team is not None else 1)
        else:
            team_ball_control.append(default_team if default_team is not None else 1)
    
    team_ball_control = np.array(team_ball_control)

    # Draw output
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
    output_video_frames = speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)
    
    # Save video with correct frame rate
    height, width = output_video_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_videos/output_video.avi', 
                         fourcc, 
                         fps,  # Use original fps
                         (width, height))
                         
    for frame in output_video_frames:
        out.write(frame)
    out.release()

if __name__ == '__main__':
    main()

# def main():
   
#         # Read Video
#         video_frames  = read_video('input_videos/08fd33_4.mp4')
#         print(f"Number of frames read: {len(video_frames)}")
#         if len(video_frames) == 0:
#             print("No frames were read from the video.")
        
#         # Initalize Tracker
#         tracker = Tracker('models/best.pt')

#         tracks = tracker.get_object_tracks(video_frames,
#                                             read_from_stub=True, 
#                                             stub_path='stubs/track_stubs.pkl')
        
#         # camera movement estimator
#         camera_movement_estimator = CameraMovementEstimator(video_frames[0])
#         camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
#                                                                                 read_from_stub=True,
#                                                                                 stub_path='stubs/camera_movement_stub.pkl')
#         camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)

#         # View Trasnformer
#         view_transformer = ViewTransformer()
#         view_transformer.add_transformed_position_to_tracks(tracks)


#         # Interpolate Ball Positions
#         tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

#         # Get object positions 
#         tracker.add_position_to_tracks(tracks)        
        
#         # Assign Player Teams
#         team_assigner = TeamAssigner()
#         team_assigner.assign_team_color(video_frames[0], 
#                                         tracks['players'][0])
    
#         for frame_num, player_track in enumerate(tracks['players']):
#                 for player_id, track in player_track.items():
#                         team = team_assigner.get_player_team(video_frames[frame_num],   
#                                                                 track['bbox'],
#                                                                 player_id)
#                         tracks['players'][frame_num][player_id]['team'] = team 
#                         tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

#          # Speed and distance estimator
#         speed_and_distance_estimator = SpeedAndDistance_Estimator()
#         speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)
        

#         # Assign Ball Aquisition
#         player_assigner =PlayerBallAssigner()
#         team_ball_control= []
#         for frame_num, player_track in enumerate(tracks['players']):
#                 ball_bbox = tracks['ball'][frame_num][1]['bbox']
#                 assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

#                 if assigned_player != -1:
#                    tracks['players'][frame_num][assigned_player]['has_ball'] = True
#                    team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
#                 else:
#                      team_ball_control.append(team_ball_control[-1])
#         team_ball_control= np.array(team_ball_control)    

#         # Drew Output
#         ## Drew Object Tracks
#         output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
        
#         ## Draw Camera movement
#         output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)

#         ## Draw Speed and Distance
#         speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)

#         # Save Video    
#         save_video(output_video_frames, 'output_videos/output_video.avi')
        
# if __name__ == '__main__':
#     main()