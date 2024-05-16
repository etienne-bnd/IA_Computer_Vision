import cv2
import numpy as np
from framebyframe import framebyframe
# from imageStitching import image_stitcher

def get_image_halves(image):
    """
    Cette fonction prend une image en entrée et renvoie la partie gauche et la partie droite de l'image.
    
    Args:
    - image (numpy.ndarray): L'image d'entrée.
    
    Returns:
    - left_half (numpy.ndarray): La partie gauche de l'image.
    - right_half (numpy.ndarray): La partie droite de l'image.
    """
    # Obtenir la largeur de l'image
    height, width = image.shape[:2]
    
    # Définir la largeur de chaque moitié (la moitié de la largeur de l'image)
    half_width = width // 2
    
    # Extraire la partie gauche de l'image
    left_half = image[:, :half_width]
    
    # Extraire la partie droite de l'image
    right_half = image[:, half_width:]
    
    return left_half, right_half


def get_image_halves_without_border(image):
    """
    Cette fonction prend une image en entrée et renvoie la partie gauche et la partie droite de l'image.
    sans les gradins
    
    Args:
    - image (numpy.ndarray): L'image d'entrée.
    
    Returns:
    - left_half (numpy.ndarray): La partie gauche de l'image.
    - right_half (numpy.ndarray): La partie droite de l'image.
    """
        # on coupe en deux pour recoller après
    left_half, right_half = get_image_halves(image)

    # on enlève les gradins pour analyser le terrain
    _, left_half = get_image_halves(left_half)
    right_half, _ = get_image_halves(right_half)

    return left_half, right_half





if __name__ == "__main__":


    video_path = "videos_out_reserve//out10.mp4"
    image = framebyframe(video_path, 0)
    if image is None:
        print(f"Impossible de charger l'image ")
        exit()
    # Obtenir les parties gauche et droite de l'image
    left_half, right_half = get_image_halves_without_border(image)

    cv2.imwrite("left_part.png",left_half) 
    cv2.imwrite("right_part.png",right_half) 

    



