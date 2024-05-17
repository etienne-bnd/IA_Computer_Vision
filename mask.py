import cv2
import numpy as np
from resize_imageP import resize_image

def NCC(image1, image2):
    """ pour calculer le décalage des images c'est peut-être ça qui cause un problème dans le stitching"""

    # Trouver le décalage entre les images en utilisant la corrélation croisée normalisée (NCC)
    result = cv2.matchTemplate(image1, image2, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Trouver le décalage
    dx = max_loc[0] - min_loc[0]
    dy = max_loc[1] - min_loc[1]

    # Afficher le décalage
    print("Décalage entre les images : dx =", dx, ", dy =", dy)


def create_mask(image, position='left', border_ratio=0.01):
    """
    Crée un masque pour l'image spécifiée en laissant une marge définie par border_ratio.

    Args:
    - image (numpy.ndarray): L'image d'entrée.
    - position (str): 'left' pour créer un masque pour la partie gauche de l'image,
                      'right' pour la partie droite.
    - border_ratio (float): Le ratio de la largeur de l'image qui doit être laissé comme marge.

    Returns:
    - mask (numpy.ndarray): Le masque créé.
    """
    height, width = image.shape[:2]
    border_width = int(width * border_ratio)
    mask = np.zeros((height, width), dtype=np.uint8)
    #normalement vrai truc mais je teste autre chose
    if position == 'right':
        mask[:, :width // 7] = 255
    elif position == 'left':    
        mask[:, 6 * width // 7 :] = 255

    # if position == 'left':
    #     mask[:, :10 * width // 11] = 255
    # elif position == 'right':    
    #     mask[:, width // 11 :] = 255
    
    return mask

def main():
    # Charger les images
    image_left = cv2.imread('left_part.png')
    # image_left = resize_image(image_left)
    image_right = cv2.imread('right_part.png')
    # image_right = resize_image(image_right)
    # NCC(image_left, image_right)
    # cv2.imshow("left", image_left)
    # cv2.imshow("right", image_right)
    # cv2.waitKey(0)

    if image_left is None or image_right is None:
        print("Erreur lors du chargement des images")
        return

    # Créer des masques pour les images
    mask_left = create_mask(image_left, 'left', border_ratio=0.01)
    mask_right = create_mask(image_right, 'right', border_ratio=0.01)


    ### pour voir les masques ###
    # # Superposer les masques sur les images
    # image_left_with_mask = cv2.bitwise_and(image_left, image_left, mask=mask_left)
    # image_right_with_mask = cv2.bitwise_and(image_right, image_right, mask=mask_right)

    # # Afficher les images avec les masques
    # cv2.imshow("Left Image with Mask", image_left_with_mask)
    # cv2.imshow("Right Image with Mask", image_right_with_mask)
    # cv2.waitKey(0)
    images = [image_left, image_right]
    masks = [mask_left, mask_right]

    # Créer l'image Stitcher
    imageStitcher = cv2.Stitcher_create(mode=cv2.Stitcher_SCANS)
    seuil_confiance = 0.6 # Par exemple, 1%
    imageStitcher.setPanoConfidenceThresh(seuil_confiance)
    # Exécuter la fonction stitch avec les masques
    # Définir le mode souhaité
    # nouveau_mode = cv2.Stitcher_PANORAMA  # Remplacez Stitcher_SCANS par le mode que vous souhaitez utiliser

    # # Définir le mode sur le Stitcher
    # imageStitcher.setInterpolationFlags(nouveau_mode)
    status, stitched_img = imageStitcher.stitch(images, masks)

    # Vérifier si le stitching s'est déroulé avec succès
    if status == cv2.Stitcher_OK:
        print("Stitching réussi")
        # Récupérer lamatrice de transformation
        # mask = imageStitcher.resultMask()
        # cv2.imwrite('mask.png', mask)

        cv2.imwrite("stitchedOutput.png", stitched_img)
        stitched_img = resize_image(stitched_img, 20)
        cv2.imshow("Stitched Image", stitched_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Le stitching a échoué : ", status)

if __name__ == "__main__":
    
    main()
