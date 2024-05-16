import numpy as np
import cv2
from get_image_halves import get_image_halves
from framebyframe import framebyframe
from resize_imageP import resize_image
from mask import create_mask
from get_image_halves import get_image_halves





def image_stitcher(images, masks=None, seuil_confiance=0.1):
    imageStitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
        # Définir les indicateurs d'interpolation
    # imageStitcher.setInterpolationFlags(cv2.INTER_NEAREST)
     # Par exemple, 1%
    imageStitcher.setPanoConfidenceThresh(seuil_confiance)
    error, stitched_img = imageStitcher.stitch(images, masks)

    if not error:

        cv2.imwrite("stitchedOutput.png", stitched_img)
        stitched_img = resize_image(stitched_img, 20)
        cv2.imshow("Stitched Img", stitched_img)
        cv2.waitKey(0)
        return stitched_img

    else:
        print("Images could not be stitched!")
        print("Likely not enough keypoints being detected!")
        print(error)
        return None
    






if __name__ == "__main__":
    video_path = "videos_out_reserve//out10.mp4"
    frame = framebyframe(video_path, 0)
    left_image, right_image = get_image_halves(frame)
    left_path = "left_part.png"
    right_path = "right_part.png"

    left_mask = create_mask(left_image, 'left')
    right_mask = create_mask(right_image, 'right')
    left_image = cv2.imread(left_path)
    right_image = cv2.imread(right_path)
    images = [left_image, right_image]
    masks = [left_mask, right_mask]
    image_stitcher(images, masks)


