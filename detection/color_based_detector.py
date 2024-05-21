import cv2
import numpy as np
from detection.utils_detection import reshape_box, test_oob, sort_players
from random import shuffle


    
class ColorBased_ObjectDetection:
    def __init__(self,
                 n_objects,
                 lower_mask=np.array([0, 0, 0]),
                 upper_mask= np.array([100, 100, 100]),
                 bounds=None,
                 reshape=True,
                 ):
        """
        Initializes an instance of color-based object detection.

        Args:
            n_objects (int): The number of objects to detect.
            lower_mask (numpy.ndarray): The lower bound of the color mask in BGR format (default [0, 0, 0]).
            upper_mask (numpy.ndarray): The upper bound of the color mask in BGR format (default [100, 100, 100]).
            bounds (tuple, optional): The bounds (x, y, width, height) to restrict detection to a specific region.
            reshape (bool): Indicates whether the image should be resized (default True).

        Attributes:
            lower_mask (numpy.ndarray): The lower bound of the color mask.
            upper_mask (numpy.ndarray): The upper bound of the color mask.
            bounds (tuple): The bounds to restrict detection to a specific region.
            reshape (bool): Indicates whether the image should be resized.
            n_objects (int): The number of objects to detect.
        """
        self.lower_mask = lower_mask  # Sets the lower bound of the color mask
        self.upper_mask = upper_mask  # Sets the upper bound of the color mask
        self.bounds = bounds  # Sets the bounds to restrict detection to a specific region
        self.reshape = reshape  # Indicates whether the image should be resized
        self.n_objects = n_objects  # Sets the number of objects to detect


    def detect(self, frame):
        """
        Detect objects in a given frame based on a predefined color range.

        Args:
            frame (numpy.ndarray): The input image frame.

        Returns:
            list: Sorted bounding boxes of detected objects if successful.
            None: If detection fails.
        """
        # Create mask based on the color range
        mask = cv2.inRange(frame, self.lower_mask, self.upper_mask)
        
        # Apply mask to the frame
        masked_image = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Convert masked image to grayscale
        masked_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)

        # Define ranges for erosion and dilation
        erosions = [i for i in range(10, 30, 3)]
        dilations = [i for i in range(20, 30, 5)]

        bounding_boxes = []

        for erosion in erosions:
            for dilation in dilations:
                # Erode and dilate to remove noise
                masked_image = cv2.erode(masked_image, kernel=np.ones((erosion, erosion), np.uint8), iterations=1)
                masked_image = cv2.dilate(masked_image, kernel=np.ones((dilation, dilation), np.uint8), iterations=1)
                
                # Apply binary threshold
                _, binary_image = cv2.threshold(masked_image, 1, 255, cv2.THRESH_BINARY)
                
                # Find contours in the binary image
                contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                bounding_boxes = []
                
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    if self.reshape:
                        (x, y, w, h) = reshape_box((x, y, w, h))
                        if test_oob((x, y, w, h), self.bounds):
                            bounding_boxes.append((x, y, w, h))
                
                # Print the current erosion level and the number of bounding boxes detected
                print(erosion, len(bounding_boxes))
                
                # Check if the number of detected objects matches the expected number
                if len(bounding_boxes) == self.n_objects:
                    self.erosion = erosion
                    self.dilation = dilation
                    return sort_players(bounding_boxes)

        # If detection fails, print a message and return None
        print("detection failed")
        return None
