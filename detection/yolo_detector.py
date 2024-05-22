import cv2
import numpy as np
import sys
import os

# Ajouter le répertoire racine au PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from detection.utils_detection import * 

class Yolo_ObjectDetection:
    def __init__(self, nmsThreshold = 0.5, confThreshold = 0.5, weights_path="detection/yolo_model/yolov4.weights", cfg_path="detection/yolo_model/yolov4.cfg",
                 bounds=None,
                 reshape=True):
        """
        Initializes an instance of the ObjectDetection class with YOLO inference capabilities.

        Args:
            nmsThreshold (float): The Non-Maximum Suppression (NMS) threshold for post-processing detections (default is 0.5).
            confThreshold (float): The confidence threshold for filtering detections (default is 0.5).
            weights_path (str): The file path to the YOLO weights file (default is "detection/yolo_model/yolov4.weights").
            cfg_path (str): The file path to the YOLO configuration file (default is "detection/yolo_model/yolov4.cfg").
            bounds (tuple, optional): The bounds (xmin, xmax, ymin, ymax) to restrict detection to a specific region.
            reshape (bool): Indicates whether the input image should be reshaped to a square (default is True).

        This function initializes an ObjectDetection instance with YOLO inference capabilities.
        It loads the YOLO network for object detection, sets preferable backend and target to CUDA for GPU acceleration,
        and sets up parameters such as NMS threshold, confidence threshold, image size, and input preprocessing.
        Optionally, it allows specifying bounds to restrict detection to a specific region and whether to reshape input images.
        """
        self.reshape=reshape
        self.bounds=bounds

        # Load Network detection
        self.nmsThreshold = nmsThreshold
        self.confThreshold = confThreshold
        self.image_size = 608
        net = cv2.dnn.readNet(weights_path, cfg_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.model = cv2.dnn_DetectionModel(net)
        self.classes = []
        self.load_class_names()
        self.model.setInputParams(size=(self.image_size, self.image_size), scale=1/255)

    def load_class_names(self, classes_path="detection/yolo_model/classes.txt"):
        """
        Loads class names from a text file.

        Args:
            classes_path (str): The file path to the text file containing class names (default is "detection/yolo_model/classes.txt").

        Returns:
            list: A list containing the loaded class names.

        This method reads class names from the specified text file, strips any leading or trailing whitespace,
        and appends them to the class list attribute of the ObjectDetection instance.
        """
        with open(classes_path, "r") as file_object:
            for class_name in file_object.readlines():
                class_name = class_name.strip()
                self.classes.append(class_name)
        return self.classes

    def detect(self, frame):
        """
        Performs object detection on a given frame.

        Args:
            frame (numpy.ndarray): The input image frame on which detection is to be performed.

        Returns:
            list: A list containing bounding boxes of detected objects, sorted by their positions.

        This method utilizes the YOLO model to perform object detection on the input frame.
        It filters the detected bounding boxes based on whether they lie within the specified bounds (if provided)
        and whether the reshaping of boxes is enabled.
        Finally, it returns the sorted list of bounding boxes of detected objects.
        If no players are found, it raises an assertion error.
        """
    
        _,_,boxes=self.model.detect(frame, nmsThreshold=self.nmsThreshold, confThreshold=self.confThreshold)
        bounding_boxes=[]
        for box in boxes:
            if self.reshape:
                (x, y, w, h)=reshape_box(box)
                if test_oob((x, y, w, h),self.bounds):
                    bounding_boxes.append((x, y, w, h))

        assert len(bounding_boxes)>0,"Detection failed, no players were found"
  

        return sort_players(bounding_boxes)
