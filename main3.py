from player_ball_assignment import PlayerBallAssigner
from utils import read_video, save_video
from team_assigner import TeamAssigner
from trackers import Tracker

import cv2
import numpy as np

def main():
    video_frames = read_video('input_videos/08fd33_4.mp4')
    
    tracker = Tracker('models/best.pt')
    
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    
    
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks["players"][0])
    
    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
            
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)
    
    #draw output
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    
    
    #save result
    save_video(output_video_frames, 'output_videos/output_video_test40.avi')

if __name__ == '__main__':
    main()