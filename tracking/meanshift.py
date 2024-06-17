import cv2
import numpy as np
from matplotlib import pyplot as plt

class Hist_Tracker:
    def __init__(self,frame,box,tracker_id):

        """ 
        A tracker is associated to a player: it is initialized by analysing the histogram of the bounding box representing the player
        When calling tracker.find_box(frame), the bounding box (self.current_box) is updated with the meanshift tracking, the speed is updated
        """

        # initialisation
        (x, y, w, h)=box
        # Set up the ROI (region of interest) for tracking
        roi = frame[y:y+h, x:x+w]

        #test
        mask_maillot1= cv2.inRange(roi, np.array([0,0,0]), np.array([100,100,100]))
        mask_maillot2= cv2.inRange(roi, np.array([200,200,200]), np.array([255, 255, 255]))

        
        mask=cv2.bitwise_or(mask_maillot1,mask_maillot2)
        plt.imshow(mask, cmap='gray')
        roi_hist = cv2.calcHist([roi],[0,1,2],mask,[3,3,3],[0,255,0,255,0,255])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        # we saved the data which will help us to find the roi in the next frame
        self.term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1) #controls tracking
        self.roi_hist=roi_hist
        self.current_box=box
        self.speed=0.3 #initial speed value
        self.tracker_id=tracker_id

 

    def find_box(self,frame):
        """method for finding the new box of the previous object"""
        # the projected histogram is calculated to find the object by using the roi

        dst = cv2.calcBackProject([frame],[0,1,2],self.roi_hist,[0,255,0,255,0,255],1)
        dst = cv2.erode(dst,kernel=np.ones((2,2)))
        cv2.namedWindow("dst", cv2.WINDOW_NORMAL) 
        cv2.imshow('dst', dst)
        cv2.waitKey(1)
        ret, new_box = cv2.meanShift(dst, self.current_box, self.term_crit)
        if ret:
            inertia_factor=0.8
            d=((new_box[0]-self.current_box[0])**2+2*(new_box[1]-self.current_box[1])**2)**0.5 #distance travelled by the player between the 2 frames
            if d/(1*self.speed+1)<30: #disallow the biggest displacements
                self.speed=round(inertia_factor*self.speed+(1-inertia_factor)*d/10,1)
                self.current_box = new_box
        else:
            print("Mean Shift Warning: find box failed")