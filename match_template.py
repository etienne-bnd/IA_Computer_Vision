import cv2
import numpy as np

# Charger l'image principale et le modèle
main_image = cv2.imread('left_part.png', cv2.IMREAD_GRAYSCALE)
template = cv2.imread('right_part.png', cv2.IMREAD_GRAYSCALE)

# Trouver le modèle dans l'image principale en utilisant la corrélation de modèle
result = cv2.matchTemplate(main_image, template, cv2.TM_CCOEFF_NORMED)

# Définir un seuil de confiance pour la correspondance
threshold = 0.8

# Trouver les coordonnées où le modèle a été trouvé avec une confiance supérieure au seuil
locations = np.where(result >= threshold)
locations = list(zip(*locations[::-1]))

# Dessiner des rectangles autour des correspondances trouvées
for loc in locations:
    bottom_right = (loc[0] + template.shape[1], loc[1] + template.shape[0])
    cv2.rectangle(main_image, loc, bottom_right, 255, 2)

# Afficher l'image avec les correspondances trouvées
cv2.imshow('Matching Result', main_image)
cv2.imwrite("match.png", main_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
