import cv2
from framebyframe import framebyframe
from resize_imageP import resize_image
import tkinter as tk


def get_image_halves(image):
    """
    This function takes an image as input and returns the left and right halves of the image.
    
    Args:
    - image (numpy.ndarray): The input image.
    
    Returns:
    - left_half (numpy.ndarray): The left half of the image.
    - right_half (numpy.ndarray): The right half of the image.
    """
    # Get the width of the image
    _, width = image.shape[:2]
    
    # Define the width of each half (half of the image width)
    half_width = width // 2
    
    # Extract the left half of the image
    left_half = image[:, :half_width - 2]
    
    # Extract the right half of the image
    right_half = image[:, half_width + 2:]
    
    return left_half, right_half



def get_image_halves_without_border(image):
    """
    This function takes an image as input and returns the left and right halves of the image.
    without borders.

    Args:
    - image (numpy.ndarray): The input image.

    Returns:
    - left_half (numpy.ndarray): The left half of the image.
    - right_half (numpy.ndarray): The right half of the image.
    """
    # Splitting the image into halves for rejoining later
    left_half, right_half = get_image_halves(image)

    # Removing borders for terrain analysis
    _, left_half = get_image_halves(left_half)
    right_half, _ = get_image_halves(right_half)

    return left_half, right_half



def get_screen_size():
    """Get the screen size in pixels.

    Returns:
        tuple: Screen width and height in pixels.
    """

    root = tk.Tk()
    root.withdraw()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    return screen_width, screen_height


if __name__ == "__main__":
    video_path = "stitching//videos_out_reserve//out10.mp4"
    image = framebyframe(video_path, 0)
    if image is None:
        print(f"Impossible to load the image ")
        exit()
    # Get the left and right halves of the image
    left_half, right_half = get_image_halves_without_border(image)
    cv2.imwrite("left_part.png", left_half) 
    cv2.imwrite("right_part.png", right_half) 


    left_half = resize_image(left_half, 20)
    right_half = resize_image(right_half, 20)

    screen_width, screen_height = get_screen_size()

    # Calculate coordinates to center the windows
    left_x = max(0, (screen_width - left_half.shape[1]) // 2)
    left_y = max(0, (screen_height - left_half.shape[0]) // 2)
    right_x = max(0, left_x + left_half.shape[1])
    right_y = left_y

    # Display images at the center of the screen
    cv2.imshow("left half", left_half)
    cv2.moveWindow("left half", left_x, left_y)
    cv2.imshow("right half", right_half)
    cv2.moveWindow("right half", right_x, right_y)

    # Wait for a key press
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    



