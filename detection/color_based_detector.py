import cv2
import numpy as np
from detection.utils_detection import reshape_box, test_oob, sort_players
from random import shuffle


    
class ColorBased_ObjectDetection:
    def __init__(self,
                 n_objects,
                 lower_mask=np.array([0, 0, 0]),
                 upper_mask= np.array([100,100,100]),
                 bounds=None,
                 reshape=True,
                 ):
        self.lower_mask = lower_mask
        self.upper_mask = upper_mask
        self.bounds = bounds
        self.reshape = reshape
        self.n_objects=n_objects

    def detect(self, frame):
        # Create mask based on the color range
        mask = cv2.inRange(frame, self.lower_mask, self.upper_mask)
        
        # Apply mask to the frame
        masked_image = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Convert masked image to grayscale
        masked_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
        erosions=[i for i in range(10,30,3)]
        dilations=[i for i in range(5,30,5)]
        # shuffle(erosions)
        bounding_boxes = []
        for erosion in erosions:
            for dilation in dilations:
                # Erode and dilate to remove noise
                masked_image = cv2.erode(masked_image, kernel=np.ones((erosion,erosion), np.uint8), iterations=1)
                masked_image = cv2.dilate(masked_image, kernel=np.ones((dilation, dilation), np.uint8), iterations=1)
                # Apply binary threshold
                _, binary_image = cv2.threshold(masked_image, 1, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                bounding_boxes = []
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    if self.reshape:
                        (x, y, w, h) = reshape_box((x, y, w, h))
                        if test_oob((x, y, w, h), self.bounds):
                            bounding_boxes.append((x, y, w, h))
                print(erosion,len(bounding_boxes))
                if len(bounding_boxes)==self.n_objects:
                    self.erosion=erosion
                    self.dilation=dilation
                    return sort_players(bounding_boxes)

        print("detection failed")
        return(None)
