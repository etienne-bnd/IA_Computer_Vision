import cv2


def resize_image(img, scale_percent=50):
    """Resize an image for computer display.

    Args:
        img (numpy.ndarray): The input image to be resized.
        scale_percent (int, optional): The percentage scale for resizing the image. Default is 50.

    Returns:
        numpy.ndarray: The resized image.
    """
    
   # Calculate new width and height based on the scale
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    # Resize the image
    resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return resized_img