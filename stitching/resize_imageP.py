# Fonction pour redimensionner une image pour qu'elle s'adapte à l'écran
import cv2


def resize_image(img, scale_percent=50):
    """function for sizing image at the computer display"""
    # Calculer la nouvelle largeur et hauteur en fonction de l'échelle
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    # Redimensionner l'image
    resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return resized_img