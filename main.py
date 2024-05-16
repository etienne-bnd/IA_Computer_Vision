from test_avec_hommography import hommography, hommography_return_M, apply_the_matrix
from final_transformation import transformation
from get_image_halves import get_image_halves
from framebyframe import framebyframe
import cv2
from resize_imageP import resize_image

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

def frame_to_final_with_M(frame, M):

    # on coupe en deux pour recoller après
    left_half, right_half = get_image_halves(frame)

    # on enlève les gradins pour analyser le terrain
    _, left_half = get_image_halves(left_half)
    right_half, _ = get_image_halves(right_half)

    # on utilise l'hommography pour les rassembler
    stitched_img = apply_the_matrix(M, left_half, right_half)
    
    # on renvoie la transformation finale qui enlève les bordures noires
    return transformation(stitched_img)





if __name__ == "__main__":
    video_path = "videos_out_reserve//out10.mp4"
    output_path = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
### partie ou on calcule la matrice une fois et on réaplique à chaque passage
    img1 = cv2.imread('left_part.png')
    img2 = cv2.imread('right_part.png')
    M = hommography_return_M(img1, img2)
    frame_0 = frame_to_final_with_M(framebyframe(video_path, 0), M)
    height, width, _ = frame_0.shape
    for i in range(100):
        frame = framebyframe(video_path, i)
        frame = frame_to_final_with_M(frame, M)
        frame_affich = resize_image(frame, 20)
        cv2.imshow("final", frame_affich)
        cv2.waitKey(1)


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
