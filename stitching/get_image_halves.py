import cv2
from stitching.framebyframe import framebyframe
from stitching.resize_imageP import resize_image
import numpy as np
import tkinter as tk
from tkinter import messagebox


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


def get_screen_size():
    root = tk.Tk()
    root.withdraw()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    return screen_width, screen_height


if __name__ == "__main__":
    video_path = "videos_out_reserve//out10.mp4"
    image = framebyframe(video_path, 0)
    if image is None:
        print(f"Impossible de charger l'image ")
        exit()
    # Obtenir les parties gauche et droite de l'image
    left_half, right_half = get_image_halves_without_border(image)
    left_half = resize_image(left_half, 20)
    right_half = resize_image(right_half, 20)
    cv2.imshow("left half", left_half)
    cv2.imshow("right half", right_half)
    # Obtenir la taille de l'écran
    screen_width, screen_height = get_screen_size()

    # Calculer les coordonnées pour centrer les fenêtres
    left_x = max(0, (screen_width - left_half.shape[1]) // 2)
    left_y = max(0, (screen_height - left_half.shape[0]) // 2)
    right_x = max(0, left_x + left_half.shape[1])
    right_y = left_y

    # Afficher les images au centre de l'écran
    cv2.imshow("left half", left_half)
    cv2.moveWindow("left half", left_x, left_y)
    cv2.imshow("right half", right_half)
    cv2.moveWindow("right half", right_x, right_y)

    # Attendre une pression de touche
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite("left_part.png",left_half) 
    # cv2.imwrite("right_part.png",right_half) 

    



