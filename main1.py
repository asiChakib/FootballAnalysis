from utils import read_video, save_video
from trackers import Tracker

def main():
    video_frames = read_video('input_videos/08fd33_4.mp4')
    
    tracker = Tracker('models/best.pt')
    
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    
    #draw output
    
    output_video_frames = tracker.draw_annotations(video_frames, tracks)
    
    
    #save result
    save_video(output_video_frames, 'output_videos/output_video_test04.avi')

if __name__ == '__main__':
    main()