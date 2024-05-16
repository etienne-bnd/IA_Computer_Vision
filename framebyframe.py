import cv2

def framebyframe(video_path, frame_number):

    """
    Cette fonction extrait une frame spécifique à partir d'une vidéo.

    Args:
        video_path (str): Le chemin vers la vidéo.
        frame_number (int): Le numéro de la frame à extraire.

    Returns:
        Union[numpy.ndarray, bool]: L'image de la frame extraite si la lecture réussit, sinon False.
    """
    video_capture = cv2.VideoCapture(video_path)
    
    # Vérifier si la vidéo a pu être ouverte
    if not video_capture.isOpened():
        print(f"Impossible d'ouvrir la vidéo : {video_path}")
        return False
    
    # Aller à la frame spécifiée dans la vidéo
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Lire la frame spécifiée
    success, frame = video_capture.read()
    
    # Vérifier si la lecture de la frame a réussi
    if not success:
        print("Impossible de lire la frame spécifiée")
        return False
    
    return frame


def count_frames(video_path):
    """
    Cette fonction compte le nombre de frames dans une vidéo.

    Args:
        video_path (str): Le chemin vers la vidéo.

    Returns:
        int: Le nombre total de frames dans la vidéo, ou -1 en cas d'erreur.
    """
    
    # Ouvrir la vidéo
    video_capture = cv2.VideoCapture(video_path)
    
    # Vérifier si la vidéo a pu être ouverte
    if not video_capture.isOpened():
        print(f"Impossible d'ouvrir la vidéo : {video_path}")
        return -1  # Retourner -1 en cas d'erreur
    
    # Obtenir le nombre total de frames dans la vidéo
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Fermer la capture vidéo
    video_capture.release()
    
    return total_frames


if __name__ == "__main__":

    video_path = "videos_out_reserve//out10.mp4"
    print(count_frames(video_path))
    frame = framebyframe(video_path, 10)
    cv2.imshow("frame", frame)
    cv2.waitKey(0)


