import cv2
import imutils
import numpy as np

def transformation(stitched_img): 
    """Remove dark borders from a stitched image.

    Args:
        stitched_img (numpy.ndarray): The stitched image with dark borders.

    Returns:
        numpy.ndarray: The cropped image without dark borders.
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

    stitched_img = stitched_img[y:y + h, x:x + w]
    return stitched_img

def transformation_return_transformation(stitched_img): 
    """Remove dark borders from a stitched image and return the region of interest.

    Args:
        stitched_img (numpy.ndarray): The stitched image with dark borders.

    Returns:
        tuple: Coordinates and dimensions of the region of interest (x, y, w, h).
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
    """Crop dark borders from a stitched image.

    Args:
        list_value (list): A list containing the coordinates [x, y, w, h] of the bounding box to crop.
        stitched_img (numpy.ndarray): The stitched image containing dark borders.

    Returns:
        numpy.ndarray: The cropped image without dark borders.
    """

    [x, y, w, h] = list_value

    stitched_img = stitched_img[y:y + h, x:x + w]

    return stitched_img