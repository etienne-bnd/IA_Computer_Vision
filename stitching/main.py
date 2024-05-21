from homography import homography, homography_return_M, apply_the_matrix, homography_return_M_roi
from final_transformation import transformation, transformation_data, transformation_return_transformation
from get_image_halves import get_image_halves_without_border
from framebyframe import framebyframe, count_frames
import cv2
from resize_imageP import resize_image
from tqdm import tqdm
from mask import create_mask
from imageStitching import image_stitcher




def frame_to_final(frame):
    """Transform a frame into a final stitched image.

    Args:
        frame (numpy.ndarray): Input frame to process.

    Returns:
        numpy.ndarray: The final transformed image without black borders.
    """
    # Splitting the frame into halves for rejoining later
    left_half, right_half = get_image_halves_without_border(frame)

    # Using homography to merge them
    # stitched_img = homography(left_half, right_half)
    
    M = homography_return_M_roi(left_half, right_half)
    stitched_img = apply_the_matrix(M, left_half, right_half)

    # Return the final transformation removing black borders
    return transformation(stitched_img)


def frame_to_final_with_M(frame, M, list_value=None, turn="not first"):
    """From initial frame to final one.

    Args:
        frame (numpy.ndarray): Initial input frame.
        M (numpy.ndarray): Homography transformation matrix.
        list_value (list, optional): List containing the coordinates [x, y, w, h].
        turn (str, optional): Indicator for first or subsequent transformation.

    Returns:
        Union[list, numpy.ndarray]: Transformed image without black borders if 'turn' is not 'first',
                                     otherwise the transformed coordinates and image.
    """
    # Splitting the frame into halves for rejoining later
    left_half, right_half = get_image_halves_without_border(frame)

    # Using homography to merge them
    stitched_img = apply_the_matrix(M, left_half, right_half)
    
    # Return the final transformation removing black borders
    if turn == "not first":
        return transformation_data(list_value, stitched_img)
    else:
        [x, y, w, h] = transformation_return_transformation(stitched_img)
        return [x, y, w, h], transformation_data([x, y, w, h], stitched_img)


def frame_to_final_stitch(frame, list_value=None, turn="not first"):
    """Transform a frame into a final stitched image.

    Args:
        frame (numpy.ndarray): Initial input frame.
        list_value (list, optional): List containing the coordinates [x, y, w, h].
        turn (str, optional): Indicator for first or subsequent transformation.

    Returns:
        Union[list, numpy.ndarray]: Transformed image without black borders if 'turn' is not 'first',
                                     otherwise the transformed coordinates and image.
    """
    # Splitting the frame into halves for rejoining later
    left_half, right_half = get_image_halves_without_border(frame)
    
    # Creating masks for left and right halves
    left_mask = create_mask(left_half, 'left')
    right_mask = create_mask(right_half, 'right')
    masks = [left_mask, right_mask]
    
    # Copying halves for further processing
    left_half = left_half.copy()
    right_half = right_half.copy()

    images = [left_half, right_half]
    # Stitching images together
    stitched_img = image_stitcher(images, masks)
    
    # Return the final transformation removing black borders
    if turn == "not first":
        return transformation_data(list_value, stitched_img)
    else:
        [x, y, w, h] = transformation_return_transformation(stitched_img)
        return [x, y, w, h], transformation_data([x, y, w, h], stitched_img)




if __name__ == "__main__":
    video_path = "stitching//videos_out_reserve//out10.mp4"
    output_path = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    ### Section with stitching ###

    # frame = framebyframe(video_path, 0)
    # listframe_0, frame_0 = frame_to_final_stitch(framebyframe(video_path, 0), turn='first')
    # height, width, _ = frame_0.shape
    # out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
    # for i in tqdm(range(100)):
    #     frame = framebyframe(video_path, i)
    #     frame = frame_to_final_stitch(frame, list_value=listframe_0, turn="not first")
    # #     out.write(frame)
    # # out.release()

    #     ## to display frame by frame
    #     frame = resize_image(frame, 20)
    #     cv2.imshow("final", frame)
    #     cv2.waitKey(1)

    ### Section where we calculate the matrix once and reapply it each time

    frame = framebyframe(video_path, 78)
    ## 88 is good for out1O 56 / 72 / 87
    left_img, right_img = get_image_halves_without_border(frame)
    M = homography_return_M_roi(left_img, right_img)
    listframe_0, frame_0 = frame_to_final_with_M(frame, M, turn='first')
    height, width, _ = frame_0.shape
    out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
    height, width, _ = frame_0.shape
    for i in tqdm(range(1000)):
        frame = framebyframe(video_path, i)
        frame = frame_to_final_with_M(frame, M, list_value=listframe_0)
        out.write(frame)


        ## to display frame by frame ###
        frame_affich = resize_image(frame, 20)
        cv2.imshow("final", frame_affich)
        cv2.waitKey(10)
    out.release()


## Section where we recalculate the matrix frame by frame ###

    # frame_0 = frame_to_final(framebyframe(video_path, 0))
    # height, width, _ = frame_0.shape
    # out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
    # for i in range(100):
    #     frame = framebyframe(video_path, i)
    #     frame = frame_to_final(frame)
    #     # out.write(frame)
    #     frame = resize_image(frame, 20)
    #     cv2.imshow("final", frame)
    #     cv2.waitKey(1)
    #     print(i)
