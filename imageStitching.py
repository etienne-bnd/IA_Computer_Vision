import numpy as np
import cv2
from get_image_halves import get_image_halves
from framebyframe import framebyframe




def image_stitcher(images):
    imageStitcher = cv2.Stitcher_create()

    error, stitched_img = imageStitcher.stitch(images)

    if not error:

        cv2.imwrite("stitchedOutput.png", stitched_img)
        cv2.imshow("Stitched Img", stitched_img)
        cv2.waitKey(0)
        return stitched_img

    else:
        print("Images could not be stitched!")
        print("Likely not enough keypoints being detected!")
        print(error)
        return None
    






if __name__ == "__main__":
    left_path = "left_part.png"
    right_path = "right_part.png"
    left_image = cv2.imread(left_path)
    right_image = cv2.imread(right_path)
    images = [left_image, right_image]
    image_stitcher(images)


