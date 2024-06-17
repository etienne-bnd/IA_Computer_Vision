import cv2
import numpy as np
import math
import time
from meanshift import *
from utils_display import *
# from tracking.utils_display import *
import sys
import os
# Ajouter le répertoire racine au PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from detection import utils_detection, yolo_detector,color_based_detector
from evaluation import utils_eval,evaluate_function

def detect_once_and_track(path_to_video, 
                          detection_method="colorbased",
                          tracking_method="meanshift",
                          frame_lim=1000,
                          path_to_annotation="",
                          n_players=12):
    """
    1. Detect the players on the first frame, to initialize some "trackers" object (annotated by tracker_id=1,....,n)
    2. Use the method tracker.find_box(frame) to update the bounding box and keep the tracker_id coherent

    Args:
        display_shape, display_time : the shape and the time (ms) spent for the displayed frame
        detection_method: The detection method used for step 1 ("colorbased", or "Yolo")
        tracking_method: The tracking method used for step 1 ("meanshift", or "naive")
        frame_lim: The maximum number of step
        reshape: whether the boxes should be scaled down
        n_objects: number of players to track 
    """
    assert detection_method in ["colorbased","yolo"], "Wrong detection method name"
    assert tracking_method in ["meanshift"], "Wrong tracking method name"
    

    # initialize the counts
    cap = cv2.VideoCapture(path_to_video)
    count_frame = -1
    trackers={}
    id_to_center_points_prev_frame = {}
    desired_width = 1920  
    desired_height = 1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)


    while not count_frame >= frame_lim:

        ret, frame = cap.read()
        count_frame += 1
        
        if ret:
            #resize the image if necessary
            original_height, original_width = frame.shape[0], frame.shape[1]
            if (original_width != desired_width) or (original_height != desired_height):
                frame = cv2.resize(frame, (desired_width, desired_height))

            if count_frame==0:  # Step 1
                # Initialize the object detection 
                bounds=utils_detection.input_bounds(frame.copy())
                if detection_method=="yolo":
                    od = yolo_detector.Yolo_ObjectDetection(bounds=bounds)
                elif detection_method=="colorbased":
                    od = color_based_detector.ColorBased_ObjectDetection(bounds=bounds,n_objects=n_players)

                # Perform object detection
                boxes=od.detect(frame) # we collect the data for the detected objects

                for tracker_id,box in enumerate(boxes):
                    (x, y, w, h) = box
                    if tracking_method=="meanshift":
                        trackers[tracker_id+1] = Hist_Tracker(frame,box,tracker_id+1) # we add the histogram of the tracked object in the dictionnary tracker
                    else:
                        raise ValueError
                    cx,cy=x+w//2,y+h//2
                    id_to_center_points_prev_frame[tracker_id+1]=(cx,cy)
                
                if path_to_annotation:
                    try:
                        cumulated_loss=0
                        (df_boxes,df_player_ids) = utils_eval.load_annotations_from_csv(path_to_annotation)
                        summary_actual_boxes =  utils_eval.from_df_to_boxes(df_boxes,df_player_ids,(original_width,original_height),(desired_width, desired_height))
                        actual_boxes=summary_actual_boxes[0]
                        mapping=evaluate_function.compute_mapping(boxes,actual_boxes)
                        cumulated_loss+=evaluate_function.evaluate(mapping,boxes,actual_boxes)
                        additional_displays(frame,boxes=list(actual_boxes.values()),speeds=None,color=(0,0,255),offset=100)
                    except:
                        print( "Wrong path_to_annotation name or invalid data type")
                        raise ValueError
                    
                additional_displays(frame,boxes=boxes,speeds=[trackers[tracker_id+1].speed for tracker_id,box in enumerate(boxes)],color=(255,0,0))
                cv2.imshow("video",frame)


            else:  # Step 2
                boxes=[]
                for tracker_id,tracker in trackers.items():
                    tracker.find_box(frame)
                    (x, y, w, h) = tracker.current_box
                    boxes.append((x, y, w, h))
                    cx,cy=x+w//2,y+h//2
                    id_to_center_points_prev_frame[tracker_id]=(cx,cy)

                if path_to_annotation:                        
                    actual_boxes=summary_actual_boxes[count_frame]
                    cumulated_loss+=evaluate_function.evaluate(mapping,boxes,actual_boxes)
                    additional_displays(frame,boxes=list(actual_boxes.values()),speeds=None,color=(0,0,255),offset=100)
                # Display the frame with tracking information
                additional_displays(frame,boxes=boxes,speeds=[trackers[tracker_id+1].speed for tracker_id,box in enumerate(boxes)],color=(255,0,0))
                cv2.imshow("video",frame)

            
            if path_to_annotation:
                print("Errors made so far:", cumulated_loss)

        else: 
            break
        
    # Release resources and close windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ =="__main__":
    #detect_once_and_track("videos\output_video.mp4",n_players=12)
    detect_once_and_track("videos\Q4_top_30-60.mp4",path_to_annotation="videos\Q4_top_30-60.csv",n_players=10)