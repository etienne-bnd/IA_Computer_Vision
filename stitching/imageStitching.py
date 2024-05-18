import numpy as np
import cv2
from framebyframe import framebyframe
from resize_imageP import resize_image
from get_image_halves import get_image_halves_without_border





def image_stitcher(images, masks=None, seuil_confiance=0.1):
    imageStitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
        # Définir les indicateurs d'interpolation
    # imageStitcher.setInterpolationFlags(cv2.INTER_NEAREST)
     # Par exemple, 1%
    imageStitcher.setPanoConfidenceThresh(seuil_confiance)
    status, stitched_img = imageStitcher.stitch(images, masks)
    print(imageStitcher.waveCorrection())
        # Vérifiez si la fusion s'est bien déroulée
    if status == cv2.Stitcher_OK:
        # La fusion s'est bien déroulée, vous pouvez utiliser l'image fusionnée (stitched_img)
        cv2.imwrite("stitchedOutput.png", stitched_img)
        # stitched_img = resize_image(stitched_img, 20)
        # cv2.imshow("Stitched Image", stitched_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return stitched_img
    else:
        # La fusion a échoué, affichez le statut de l'erreur
        print("Erreur lors de la fusion :", status)


    






if __name__ == "__main__":
    video_path = "videos_out_reserve//out10.mp4"
    frame = framebyframe(video_path, 10)
    left_image, right_image = get_image_halves_without_border(frame)
    left_image = left_image.copy()
    right_image = right_image.copy()
    images = [left_image, right_image]
    result = image_stitcher(images)
    result = resize_image(result, 20)
    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

