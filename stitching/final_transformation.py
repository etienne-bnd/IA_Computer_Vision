import cv2
import imutils
import numpy as np
from stitching.resize_imageP import resize_image

def transformation(stitched_img): 

    """ fonction to remove dark border"""

    stitched_img = cv2.copyMakeBorder(stitched_img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, (0,0,0))    
    gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)
    thresh_img = cv2.threshold(gray, 0, 255 , cv2.THRESH_BINARY)[1]

    contours = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = imutils.grab_contours(contours)
    areaOI = max(contours, key=cv2.contourArea)

    mask = np.zeros(thresh_img.shape, dtype="uint8")
    x, y, w, h = cv2.boundingRect(areaOI)
    cv2.rectangle(mask, (x,y), (x + w, y + h), 255, -1)

    minRectangle = mask.copy()
    sub = mask.copy()

    while cv2.countNonZero(sub) > 150:
        minRectangle = cv2.erode(minRectangle, None)
        sub = cv2.subtract(minRectangle, thresh_img)


    contours = cv2.findContours(minRectangle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = imutils.grab_contours(contours)
    areaOI = max(contours, key=cv2.contourArea)


    x, y, w, h = cv2.boundingRect(areaOI)

    stitched_img = stitched_img[y:y + h, x:x + w]
    return stitched_img

def transformation_return_transformation(stitched_img): 

    """ fonction to remove dark border on renvoie que les valeurs d'intérêt pour essayer d'augmenter 
    la vitesse de computation 
    
    
    return x, y, w, 
    
    have to be used like that :


    stitched_img = stitched_img[y:y + h, x:x + w]
    """

    stitched_img = cv2.copyMakeBorder(stitched_img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, (0,0,0))    
    gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)
    thresh_img = cv2.threshold(gray, 0, 255 , cv2.THRESH_BINARY)[1]

    contours = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = imutils.grab_contours(contours)
    areaOI = max(contours, key=cv2.contourArea)

    mask = np.zeros(thresh_img.shape, dtype="uint8")
    x, y, w, h = cv2.boundingRect(areaOI)
    cv2.rectangle(mask, (x,y), (x + w, y + h), 255, -1)

    minRectangle = mask.copy()
    sub = mask.copy()

    while cv2.countNonZero(sub) > 150:
        minRectangle = cv2.erode(minRectangle, None)
        sub = cv2.subtract(minRectangle, thresh_img)


    contours = cv2.findContours(minRectangle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = imutils.grab_contours(contours)
    areaOI = max(contours, key=cv2.contourArea)


    x, y, w, h = cv2.boundingRect(areaOI)

    return x - 50, y - 50, w, h

def transformation_data(list_value, stitched_img):
    [x, y, w, h] = list_value

    stitched_img = stitched_img[y:y + h, x:x + w]

    return stitched_img