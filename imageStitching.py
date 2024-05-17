import numpy as np
import cv2
from get_image_halves import get_image_halves
from framebyframe import framebyframe
from resize_imageP import resize_image
from mask import create_mask
from get_image_halves import get_image_halves_without_border
from PIL import Image





def image_stitcher(images, masks=None, seuil_confiance=0.1):
    imageStitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
        # Définir les indicateurs d'interpolation
    # imageStitcher.setInterpolationFlags(cv2.INTER_NEAREST)
     # Par exemple, 1%
    imageStitcher.setPanoConfidenceThresh(seuil_confiance)
    status, stitched_img = imageStitcher.stitch(images, masks)
        
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
    frame = framebyframe(video_path, 0)
    left_image, right_image = get_image_halves_without_border(frame)
    print(f"Type de left_image : {type(left_image)}")
    print(f"Type de right_image : {type(right_image)}")
    cv2.imwrite("left_part.png",left_image) 
    cv2.imwrite("right_part.png",right_image) 
    left_path = 'left_part.png'
    right_path = 'right_part.png'
    left_mask = create_mask(left_image, 'left', border_ratio=0.01)
    right_mask = create_mask(right_image, 'right', border_ratio=0.01)
    # left_image = cv2.imread(left_path)
    # right_image = cv2.imread(right_path)

        # Obtenir les dimensions de l'image
    dimensions = left_image.shape

    # Extraire le nombre de canaux
    nombre_de_canaux = dimensions[-1]

    print("Nombre de canaux de l'image :", nombre_de_canaux)
    # left_image = cv2.cvtColor(left_image, None) 

    left_image = left_image.copy() #ssurez-vous d'utiliser le bon format de couleur

    # Convertir right_image en image cv2
    right_image = right_image.copy()
    # right_image = cv2.cvtColor(right_image, cv2.COLOR_RGB2BGR) 
    # print(True in (left_image != left_image2))
    # cv2.imshow("d", left_image)
    # cv2.imshow("d", right_image)
    images = [left_image, right_image]

    masks = [left_mask, right_mask]

    image_stitcher(images, masks, seuil_confiance=0.01)

