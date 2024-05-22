import cv2
from resize_imageP import resize_image

def framebyframe(video_path, frame_number):
    """ Cette fonction extrait une frame spécifique à partir d'une vidéo.

    Args:
        video_path (str): Le chemin vers la vidéo.
        frame_number (int): Le numéro de la frame à extraire.

    Returns:
        Union[numpy.ndarray, bool]: L'image de la frame extraite si la lecture réussit, sinon False.
    """
    video_capture = cv2.VideoCapture(video_path)
    
    # Check if the video could be opened
    if not video_capture.isOpened():
        print(f"Unable to open video: {video_path}")
        return False
    
    # Go to the specified frame in the video
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Read the specified frame
    success, frame = video_capture.read()
    
    # Check if reading the frame was successful
    if not success:
        print("Unable to read the specified frame")
        return False
    
    return frame



def count_frames(video_path):
    """
    This function counts the number of frames in a video.

    Args:
        video_path (str): The path to the video.

    Returns:
        int: The total number of frames in the video, or -1 in case of an error.
    """
    
    # Open the video
    video_capture = cv2.VideoCapture(video_path)
    
    # Check if the video could be opened
    if not video_capture.isOpened():
        print(f"Unable to open video: {video_path}")
        return -1  # Return -1 in case of an error
    
    # Get the total number of frames in the video
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Release the video capture
    video_capture.release()
    
    return total_frames



if __name__ == "__main__":

    video_path = "stitching//video_in//out10.mp4"
    print(count_frames(video_path))
    frame = framebyframe(video_path, 10)
    frame = resize_image(frame, 20)
    cv2.imshow("frame", frame)
    cv2.waitKey(0)


