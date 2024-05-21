import numpy as np
import cv2
from framebyframe import framebyframe
from resize_imageP import resize_image
from get_image_halves import get_image_halves_without_border





def image_stitcher(images, masks=None, confidence_threshold=0.1):
    """Stitch multiple images into a panoramic image.

    Args:
        images (List[numpy.ndarray]): List of input images to stitch.
        masks (List[numpy.ndarray], optional): List of masks corresponding to input images.
        confidence_threshold (float, optional): Confidence threshold for panorama formation.

    Returns:
        numpy.ndarray: The stitched panoramic image if successful, otherwise None.
    """
    imageStitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
    imageStitcher.setPanoConfidenceThresh(confidence_threshold)
    status, stitched_img = imageStitcher.stitch(images, masks)
    # Check if stitching was successful
    if status == cv2.Stitcher_OK:
        # Stitching successful, use the stitched image
        cv2.imwrite("stitchedOutput.png", stitched_img)
        # stitched_img = resize_image(stitched_img, 20)
        # cv2.imshow("Stitched Image", stitched_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return stitched_img
    else:
        # Stitching failed, print error status
        print("Error during stitching:", status)


    






if __name__ == "__main__":
    video_path = "stitching//videos_out_reserve//out10.mp4"
    frame = framebyframe(video_path, 11)
    left_image, right_image = get_image_halves_without_border(frame)
    left_image = left_image.copy()
    right_image = right_image.copy()
    images = [left_image, right_image]
    result = image_stitcher(images)
    result = resize_image(result, 20)
    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

