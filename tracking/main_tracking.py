import cv2
import numpy as np
import math
import time
from meanshift import *
from tracking.utils_display import *
import sys
import os
from detection import yolo_detector,color_based_detector,utils


def detect_once_and_track(path_to_video, 
                          detection_method="colorbased",
                          tracking_method="meanshift",
                          display_shape=(1000, 600),
                          display_time=1, 
                          frame_lim=1000):
    """
    1. Detect the players on the first frame, to initialize some "trackers" object (annotated by tracker_id=1,....,n)
    2. Use the method tracker.find_box(frame) to update the bounding box and keep the tracker_id coherent

    Args:
        display_shape, display_time : the shape and the time (ms) spent for the displayed frame
        detection_method: The detection method used for step 1 ("colorbased", or "Yolo")
        tracking_method: The tracking method used for step 1 ("meanshift", or "naive")
        frame_lim: The maximum number of step
        reshape: whether the boxes should be scaled down
    """

    
    assert detection_method in ["colorbased","yolo"], "Wrong detection method name"
    assert tracking_method in ["meanshift"], "Wrong tracking method name"
    
    
    # initialize the counts
    cap = cv2.VideoCapture(path_to_video)
    count_frame = -1
    trackers={}
    id_to_center_points_prev_frame = {}


    while not count_frame >= frame_lim:
        ret, frame = cap.read()
        count_frame += 1
            
        if ret:
            if count_frame==0:  # Step 1
                # Initialize the object detection 
                bounds=utils.input_bounds(frame.copy(),display_shape)
                if detection_method=="yolo":
                    od = yolo_detector.Yolo_ObjectDetection(bounds=bounds)
                elif detection_method=="colorbased":
                    od = color_based_detector.ColorBased_ObjectDetection(bounds=bounds)

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
                        
                additional_displays(frame,trackers)
                cv2.imshow("video",cv2.resize(frame, display_shape))
                cv2.waitKey(display_time)

            else:  # Step 2

                for tracker_id,tracker in trackers.items():
                    tracker.find_box(frame)
                    (x, y, w, h) = tracker.current_box
                    cx,cy=x+w//2,y+h//2
                    id_to_center_points_prev_frame[tracker_id]=(cx,cy)
                

                # Display the frame with tracking information
                additional_displays(frame,trackers)
                cv2.imshow("video",cv2.resize(frame, display_shape))
                cv2.waitKey(display_time)

        else: break
    # Release resources and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ =="__main__":
    detect_once_and_track("videos\output_video.mp4")