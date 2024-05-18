import cv2
import numpy as np
from resize_imageP import resize_image
from imageStitching import image_stitcher

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

    if position == 'right':
        mask[:, :width // 7] = 255
    elif position == 'left':    
        mask[:, 6 * width // 7 :] = 255

    
    return mask


if __name__ == "__main__":
      # Charger les images
    image_left = cv2.imread('left_part.png')
    image_right = cv2.imread('right_part.png')


    if image_left is None or image_right is None:
        print("Erreur lors du chargement des images")

    # Créer des masques pour les images
    mask_left = create_mask(image_left, 'left', border_ratio=0.01)
    mask_right = create_mask(image_right, 'right', border_ratio=0.01)


    ## pour voir les masques ###
    # Superposer les masques sur les images
    image_left_with_mask = cv2.bitwise_and(image_left, image_left, mask=mask_left)
    image_right_with_mask = cv2.bitwise_and(image_right, image_right, mask=mask_right)
    image_left_with_mask = resize_image(image_left_with_mask, 20)
    image_right_with_mask = resize_image(image_right_with_mask, 20)
    # Afficher les images avec les masques
    cv2.imshow("Left Image with Mask", image_left_with_mask)
    cv2.imshow("Right Image with Mask", image_right_with_mask)
    cv2.waitKey(0)
    images = [image_left, image_right]
    masks = [mask_left, mask_right]

    stitched_img = image_stitcher(images, masks)
    stitched_img = resize_image(stitched_img, 20)
    cv2.imshow("stitched with mask", stitched_img)
    cv2.waitKey(0)
