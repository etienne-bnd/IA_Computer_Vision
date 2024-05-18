from test_avec_hommography import hommography, hommography_return_M, apply_the_matrix
from final_transformation import transformation, transformation_data, transformation_return_transformation
from get_image_halves import get_image_halves, get_image_halves_without_border
from framebyframe import framebyframe, count_frames
import cv2
from resize_imageP import resize_image
from tqdm import tqdm
from mask import create_mask
from imageStitching import image_stitcher




def frame_to_final(frame):

    # on coupe en deux pour recoller après
    left_half, right_half = get_image_halves(frame)

    # on enlève les gradins pour analyser le terrain
    _, left_half = get_image_halves(left_half)
    right_half, _ = get_image_halves(right_half)

    # on utilise l'hommography pour les rassembler
    stitched_img = hommography(left_half, right_half)
    
    # on renvoie la transformation finale qui enlève les bordures noires
    return transformation(stitched_img)

def frame_to_final_with_M(frame, M, list_value=None, turn="not first"):
    """de la frame initiale à celle finale"""

    # on coupe en deux pour recoller après
    left_half, right_half = get_image_halves_without_border(frame)

    # on utilise l'hommography pour les rassembler
    stitched_img = apply_the_matrix(M, left_half, right_half)
    
    # on renvoie la transformation finale qui enlève les bordures noires
    if turn == "not first":
        return transformation_data(list_value, stitched_img)
    else:
        [x, y, w, h] = transformation_return_transformation(stitched_img)
        return [x, y, w, h] , transformation_data([x, y, w, h], stitched_img)

def frame_to_final_stitch(frame, list_value=None, turn="not first"):
    # on coupe en deux pour recoller après
    left_half, right_half = get_image_halves_without_border(frame)
    left_mask = create_mask(left_half, 'left')
    right_mask = create_mask(right_half, 'right')
    masks = [left_mask, right_mask]
    left_half = left_half.copy()
    right_half = right_half.copy()

    images = [left_half, right_half]
    stitched_img = image_stitcher(images, masks)
    if turn == "not first":
        return transformation_data(list_value, stitched_img)
    else:
        [x, y, w, h] = transformation_return_transformation(stitched_img)
        return [x, y, w, h] , transformation_data([x, y, w, h], stitched_img)



if __name__ == "__main__":
    video_path = "videos_out_reserve//out10.mp4"
    output_path = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

### partie avec le stitching ###
    frame = framebyframe(video_path, 0)
    listframe_0, frame_0 = frame_to_final_stitch(framebyframe(video_path, 0), turn='first')
    height, width, _ = frame_0.shape
    out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
    for i in tqdm(range(100)):
        frame = framebyframe(video_path, i)
        frame = frame_to_final_stitch(frame, list_value=listframe_0, turn="not first")
    #     out.write(frame)
    # out.release()

        ## pour afficher frame by frame
        frame = resize_image(frame, 20)
        cv2.imshow("final", frame)
        cv2.waitKey(1)

### partie ou on calcule la matrice une fois et on réaplique à chaque passage
    # img1 = cv2.imread('left_part.png')
    # img2 = cv2.imread('right_part.png')
    # M = hommography_return_M(img1, img2)
    # listframe_0, frame_0 = frame_to_final_with_M(framebyframe(video_path, 0), M, turn='first')
    # height, width, _ = frame_0.shape
    # out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
    # height, width, _ = frame_0.shape
    # for i in tqdm(range(1000)):
    #     frame = framebyframe(video_path, i)
    #     frame = frame_to_final_with_M(frame, M, list_value=listframe_0)
    #     out.write(frame)


        ### pour afficher frame by frame ###
        # frame_affich = resize_image(frame, 20)
        # cv2.imshow("final", frame_affich)
        # cv2.waitKey(1)
    # out.release()


### partie ou on recalcule la matrice frame by frame ###
    # frame_0 = frame_to_final(framebyframe(video_path, 0))
    # height, width, _ = frame_0.shape
    # out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
    # for i in range(100):
    #     frame = framebyframe(video_path, i)
    #     frame = frame_to_final(frame)
    #     out.write(frame)
    #     # cv2.imshow("final", frame_to_final(frame))
    #     # cv2.waitKey(0)
    # out.release()

    