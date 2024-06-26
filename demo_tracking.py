import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tracking.main_tracking import detect_once_and_track
from tracking.main_tracking import detect_once_and_track

max_time_seconds = 10
detect_once_and_track(path_to_video= "videos\Q4_top_30-60.mp4",
                        path_to_annotation="videos\\annotations\\Q4_top_30-60.csv",
                        n_players=10,max_time=max_time_seconds)

detect_once_and_track(path_to_video="videos\output_video.mp4",n_players=12,max_time=max_time_seconds)
