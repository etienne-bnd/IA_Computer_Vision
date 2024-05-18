import cv2
import numpy as np
from detection.utils import * 

class Yolo_ObjectDetection:
    def __init__(self, nmsThreshold = 0.5, confThreshold = 0.5, weights_path="detection/yolo_model/yolov4.weights", cfg_path="detection/yolo_model/yolov4.cfg",
                 bounds=None,
                 reshape=True):
        """ Fonction for initiating a member of the class ObjectDetection. This fonction a Yolo Inference """

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
        """Its a method which load the class names"""
        with open(classes_path, "r") as file_object:
            for class_name in file_object.readlines():
                class_name = class_name.strip()
                self.classes.append(class_name)
        return self.classes

    def detect(self, frame):
    
        _,_,boxes=self.model.detect(frame, nmsThreshold=self.nmsThreshold, confThreshold=self.confThreshold)
        bounding_boxes=[]
        for box in boxes:
            if self.reshape:
                (x, y, w, h)=reshape_box(box)
                if test_oob((x, y, w, h),self.bounds):
                    bounding_boxes.append((x, y, w, h))

        assert len(bounding_boxes)>0,"Detection failed, no players were found"
  

        return sort_players(bounding_boxes)
