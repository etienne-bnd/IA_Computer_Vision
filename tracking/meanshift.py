import cv2
import numpy as np


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

        #test1
        # mask_yellow = cv2.inRange(roi, np.array([150,150,0]), np.array([255,255,20]))
        # mask=mask_yellow
        # mask_blue = cv2.inRange(roi, np.array([0,0,00]), np.array([50,50,100]))
        # mask=mask_blue

        #test2
        mask_yellow = cv2.inRange(roi, np.array([150,150,0]), np.array([255,255,40]))
        mask=mask_yellow
        mask_white = cv2.inRange(roi, np.array([180,180,180]), np.array([255,255,255]))
        mask=cv2.bitwise_or(mask,mask_white)
        mask_black = cv2.inRange(roi, np.array([0,0,0]),np.array([140,140,140]))
        mask=cv2.bitwise_or(mask,mask_black)
        mask_red = cv2.inRange(roi, np.array([50,0,0]), np.array([200,100,100]))
        mask=cv2.bitwise_or(mask,cv2.bitwise_not(mask_red))

        roi_hist = cv2.calcHist([roi], [0], mask, [255], [0,255])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        # we saved the data which will help us to find the roi in the next frame
        self.term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1) #controls tracking
        self.roi_hist=roi_hist
        self.current_box=box
        self.speed=0.3 #initial speed value
        self.tracker_id=tracker_id

 

    def find_box(self,frame):
        """method for finding the new box of the previous object"""
        # the projected histogram is calculated to find the object by using the roi
        dst = cv2.calcBackProject([frame], [0], self.roi_hist, [0, 255], 1)
        cv2.namedWindow("dst", cv2.WINDOW_NORMAL) 
        cv2.resizeWindow("dst", 600, 400) 
        cv2.imshow('dst', dst)
        cv2.waitKey(10)
        ret, new_box = cv2.meanShift(dst, self.current_box, self.term_crit)
        if ret:
            inertia_factor=0.8
            d=((new_box[0]-self.current_box[0])**2+2*(new_box[1]-self.current_box[1])**2)**0.5 #distance travelled by the player between the 2 frames
            # if d/(1*self.speed+1)<100: #disallow the biggest displacements
            self.speed=round(inertia_factor*self.speed+(1-inertia_factor)*d/10,1)
            self.current_box = new_box
        else:
            print("find box failed")