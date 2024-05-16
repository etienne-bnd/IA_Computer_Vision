import cv2
import numpy as np

# taking two images for stiching
img1 = cv2.imread('left_part.png')
img2 = cv2.imread('right_part.png')

def keypoints(img1, img2):
    """detection keypoints"""
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
    # ratio test as per Lowe's paper for best matches
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)
    return pts1,pts2,matches



def hommography(img1, img2):
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
        #final width, height of stiched image
        width = img1.shape[1] + img2.shape[1]
        height = img1.shape[0] + 100
        results = cv2.warpPerspective(img2, M, (width,height))
        #appending images 2 to first
        results[0:img1.shape[0],0:img1.shape[1]] = img1


    # cv2.imshow('img',results)
    # cv2.imwrite('results1.png',results)     #for saving image
    # cv2.waitKey(0)
    return results


if __name__ == "__main__":
    # taking two images for stiching
    img1 = cv2.imread('left_part.png')
    img2 = cv2.imread('right_part.png')
    results = hommography(img1, img2)
    cv2.imshow('img',results)
    cv2.imwrite('results1.png',results)     #for saving image
    cv2.waitKey(0)
