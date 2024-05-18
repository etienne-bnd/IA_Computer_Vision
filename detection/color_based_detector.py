
import cv2
import numpy as np
from detection.utils import *

class ColorBased_ObjectDetection:
    def __init__(self,
                 lower_blue = np.array([0,0,0]),
                 upper_blue = np.array([70, 70, 155]),
                 bounds=None,
                 reshape=True):

        self.bounds=bounds
        self.lower_blue=lower_blue
        self.upper_blue=upper_blue
        self.reshape=reshape

    def detect(self, frame):
        mask = cv2.inRange(frame, self.lower_blue, self.upper_blue)
        masked_image=cv2.bitwise_and(frame, frame, mask=mask)
        masked_image=cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)   
        masked_image=cv2.erode(masked_image,kernel=np.ones((10,10),np.uint8),iterations = 1)
        masked_image=cv2.dilate(masked_image,kernel=np.ones((60,60),np.uint8),iterations = 1)
        _, binary_image = cv2.threshold(masked_image, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bounding_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if self.reshape:
                (x, y, w, h)=reshape_box((x, y, w, h))
                if test_oob((x, y, w, h),self.bounds):
                    bounding_boxes.append((x, y, w, h))
        assert len(bounding_boxes)>0,"Detection failed, no players were found"
        return sort_players(bounding_boxes)
