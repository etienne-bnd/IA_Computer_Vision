import cv2
import numpy as np
from resize_imageP import resize_image
from framebyframe import framebyframe
from get_image_halves import get_image_halves_without_border
import matplotlib.pyplot as plt


def keypoints(img1, img2):
    """Detection keypoints

    Args:
        img1 (numpy.ndarray): First image.
        img2 (numpy.ndarray): Second image.

    Returns:
        tuple: Points from image 1, points from image 2, and matches.
    """
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # FLANN parameters
    # FINDING INDEX PARAMETERS FOR FLANN OPERATORS
    des1 = np.float32(des1)
    des2 = np.float32(des2)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    pts1 = []
    pts2 = []
    good_matches = []
    # ratio test as per Lowe's paper for best matches
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
            good_matches.append(m)
    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)

        # Draw matches
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Display images side by side
    plt.figure(figsize=(20,10))
    plt.imshow(img_matches)
    plt.title('Keypoint Matches')
    plt.show()


    return pts1,pts2,good_matches

def keypoints_in_roi(img1, img2, roi1, roi2):
    """Detection keypoints in specified regions of interest (ROIs).

    Args:
        img1 (numpy.ndarray): First image.
        img2 (numpy.ndarray): Second image.
        roi1 (tuple): ROI in the first image (x, y, width, height).
        roi2 (tuple): ROI in the second image (x, y, width, height).

    Returns:
        tuple: Points from image 1, points from image 2, and matches.
    """

    sift = cv2.SIFT_create()
    sift2 = cv2.SIFT_create(nfeatures=100000000,  # Nombre maximum de keypoints à détecter. Mettez à 0 pour détecter tous les keypoints.
    contrastThreshold=0.1,  # Seuil de contraste pour la détection des keypoints.
    edgeThreshold=100,  # Seuil du gradient pour la détection des keypoints.
    sigma= 1  # Valeur du sigma pour la détection des keypoints)
    )
    # cv2.KAZE_create()
    # SITF
    
    # Define the regions of interest (ROI)
    x1, y1, w1, h1 = roi1
    x2, y2, w2, h2 = roi2
    
    img1_roi = img1[y1:y1+h1, x1:x1+w1]
    img2_roi = img2[y2:y2+h2, x2:x2+w2]
    
    # Find the keypoints and descriptors with SIFT in the ROIs
    kp1, des1 = sift.detectAndCompute(img1_roi, None)
    kp2, des2 = sift.detectAndCompute(img2_roi, None)

    kp12, des12 = sift2.detectAndCompute(img1_roi, None)
    kp22, des22 = sift2.detectAndCompute(img2_roi, None)
    kp1 = list(kp1)
    kp2 = list(kp2)
    kp12 = list(kp12)
    kp22 = list(kp22)
    des1 = list(des1)
    des2 = list(des2)
    des22 = list(des22)
    des12 = list(des12)
    # kp1.extend(kp12)
    # kp2.extend(kp22)
    # des1.extend(des12)
    # des2.extend(des22)

    # kp1 = kp12
    # kp2 = kp22
    # des1 = des12
    # des2 = des22
    ### partie ou on essaie d'implémenter une détection par ligne ###
    # # Convertir les tuples en listes
    # kp1 = list(kp1)
    # kp2 = list(kp2)
    # print(kp1[0].pt[1])
    # #     # Créer un détecteur de lignes LSD
    # lsd = cv2.createLineSegmentDetector()

    # # # Détecter les lignes dans l'image
    # img1_gray = cv2.cvtColor(img1_roi, cv2.COLOR_BGR2GRAY)
    # lines1, _, _, _ = lsd.detect(img1_gray)
    # # # Créer une nouvelle liste pour stocker les nouveaux keypoints
    # # new_keypoints = []

    # # # Créer une liste de keypoints le long des lignes détectées
    # for line in lines1:
    #     for x1, y1, x2, y2 in line:
    # #         # Ajouter un keypoint au milieu de chaque ligne
    #         kp1.append(cv2.KeyPoint((x1 + x2) / 2, (y1 + y2) / 2, size=10))

    # # # Ajouter les nouveaux keypoints à la liste existante kp1
    # # kp1.extend(new_keypoints)
    # img2_gray = cv2.cvtColor(img2_roi, cv2.COLOR_BGR2GRAY)
    # lines2, _, _, _ = lsd.detect(img2_gray)
    # # # Créer une nouvelle liste pour stocker les nouveaux keypoints
    # # new_keypoints = []

    # # # Créer une liste de keypoints le long des lignes détectées
    # for line in lines2:
    #     for x1, y1, x2, y2 in line:
    # #         # Ajouter un keypoint au milieu de chaque ligne
    #         kp2.append(cv2.KeyPoint((x1 + x2) / 2, (y1 + y2) / 2, size=10))

    ### pour afficher les keypoints pour le rapport


    img1_with_keypoints = cv2.drawKeypoints(img1_roi, kp1, None)
    img2_with_keypoints = cv2.drawKeypoints(img2_roi, kp2, None)
    # img1_with_keypoints = resize_image(img1_with_keypoints, 40)
    # img2_with_keypoints = resize_image(img2_with_keypoints, 40)
    # Display the images with keypoints
    cv2.imshow('Image 1 with Keypoints', img1_with_keypoints)
    cv2.imshow('Image 2 with Keypoints', img2_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    ### ###


    # Adjust keypoint coordinates relative to the original image
    for kp in kp1:
        kp.pt = (kp.pt[0] + x1, kp.pt[1] + y1)
    for kp in kp2:
        kp.pt = (kp.pt[0] + x2, kp.pt[1] + y2)
    
    # FLANN parameters
    des1 = np.float32(des1)
    des2 = np.float32(des2)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=10)
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    # # Créer un objet BFMatcher
    # bf = cv2.BFMatcher(cv2.NORM_L2)

    # # Trouver les correspondances entre les descripteurs
    # matches = bf.knnMatch(des1, des2, k=2)


    pts1 = []
    pts2 = []
    good_matches = []
    
    # Ratio test as per Lowe's paper for best matches
    moyenne_dist = 0
    nb_match = 0
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7* n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
            good_matches.append(m)
            moyenne_dist += m.distance
            nb_match += 1
    print(moyenne_dist / nb_match)
    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)
    

    ### pour afficher les matchs ###

    # Draw matches
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Display images side by side
    plt.figure(figsize=(20,10))
    plt.imshow(img_matches)
    plt.title('Keypoint Matches in ROI')
    plt.show()
    
    return pts1, pts2, matches


def homography(img1, img2):
    """take two image and return the matching image"""
        #find correspondence
    pts1, pts2, matches = keypoints(img1, img2)
    #threshold num of correspondence obtain
    if len(matches) <= 15:
        print("image pair is not suaitable for stiching")
        #M = np.identity(3) # no homography generated
    else:
        # find homography matrix between images :
        M , mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, ransacReprojThreshold = 3)

                # Définissez les options pour la fonction d'optimisation de LMA
        # criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10000000, 1e-10000)

        # # Appliquez l'optimisation LMA pour raffiner l'homographie initiale
        # ret, M = cv2.findHomography(pts2, pts1, method=cv2.LMEDS, criteria=criteria)
        # M, mask = cv2.findHomography(pts2, pts1)
        #final width, height of stiched image
        width = img1.shape[1] + img2.shape[1]
        height = img1.shape[0] + img1.shape[0]
        results = cv2.warpPerspective(img2, M, (width,height))
        #appending images 2 to first
        results[0:img1.shape[0],0:img1.shape[1]] = img1


    # cv2.imshow('img',results)
    # cv2.imwrite('results1.png',results)     #for saving image
    # cv2.waitKey(0)
    return results

def homography_return_M(img1, img2):
    """take two image and return the matching image"""
        #find correspondence
    pts1, pts2, matches = keypoints(img1, img2)
    #threshold num of correspondence obtain
    if len(matches) <= 15:
        print("image pair is not suaitable for stiching")
        return None
        #M = np.identity(3) # no homography generated
    else:
        # find homography matrix between images :
        M , mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, ransacReprojThreshold = 3)

        return M
    



def homography_return_M_roi(img1, img2):
    """take two image and return the matching image"""
        #find correspondence
    height1, width1, _ = img1.shape
    height2, width2, _ = img2.shape

    roi1 = 5 * width1 // 6, 0, width1 // 6 , height1
    roi2 = 0, 0, width2 // 6, height2 # ROI in the second image
    pts1, pts2, matches = keypoints_in_roi(img1, img2, roi1, roi2)
    #threshold num of correspondence obtain
    if len(matches) <= 15:
        print("image pair is not suaitable for stiching")
        return None
        #M = np.identity(3) # no homography generated
    else:
        # find homography matrix between images :

        # cet algo est bien plus puissant
        M , mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, ransacReprojThreshold = 3)
        # M, _ = cv2.findHomography(pts2, pts1, method=cv2.REGRESSION)

        # M, mask = cv2.findHomography(pts2, pts1)

        # Estimation initiale de l'homographie
    print(M)
    return M


def apply_the_matrix(M, img1, img2):
        width = img1.shape[1] + img2.shape[1]
        height = img1.shape[0] + img2.shape[0]
        results = cv2.warpPerspective(img2, M, (width,height))
        #appending images 2 to first
        results[0:img1.shape[0],0:img1.shape[1]] = img1
        return results

if __name__ == "__main__":
    video_path = "stitching//videos_out_reserve//out10.mp4"
    image = framebyframe(video_path, 78)


    # 3000 à l'air bien pour récupérer la matrice pour out10
    # 9 est bien pour la matrice prour out11
    

    if image is None:
        print(f"Impossible de charger l'image ")
        exit()
        # Obtenir les parties gauche et droite de l'image
    left_half, right_half = get_image_halves_without_border(image)



    # height1, _, _ = left_half.shape
    # height2, _, _ = right_half.shape
    # left_half_up = left_half[:height1//2, :]
    # left_half_down = left_half[height2//2:, :]
    # right_half_up = right_half[:height1//2, :]
    # right_half_down = right_half[height2//2:, :]
    # M = hommography_return_M_roi(left_half_down, right_half_down)
    # down = apply_the_matrix(M, left_half_down, right_half_down)
    # M = hommography_return_M_roi(left_half_up, right_half_up)
    # up = apply_the_matrix(M, left_half_up, right_half_up)
    # results = np.vstack((up, down))


    ### avec la technique en séparant la matrice de base ###
    M = hommography_return_M_roi(left_half, right_half)
    results = apply_the_matrix(M, left_half, right_half)
    cv2.imwrite("visualisation.png", results)


    ### pour l'affichage du résultat ###
    results = resize_image(results, 50)
    cv2.imshow('img',results)
    cv2.waitKey(0)