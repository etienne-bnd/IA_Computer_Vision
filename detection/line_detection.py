import cv2
import numpy as np
 

cap = cv2.VideoCapture('videos\output_video.mp4')
ret, image = cap.read()
image = cv2.resize(image, (1920, 1080))

mask = cv2.inRange(image, np.array([120,0,0]), np.array([190,105,105]))

# Convert image to grayscale
gray = cv2.cvtColor(cv2.bitwise_and(image,image, mask=mask),cv2.COLOR_BGR2GRAY)
 
# Use canny edge detection
edges = cv2.Canny(gray, 50, 150, apertureSize=3,L2gradient=True)
 
# Apply HoughLinesP method to 
# to directly obtain line end points
lines_list =[]
lines = cv2.HoughLinesP(
            edges, # Input edge image
            1, # Distance resolution in pixels
            np.pi/180, # Angle resolution in radians
            threshold=30, # Min number of votes for valid line
            minLineLength=500, # Min allowed length of line
            maxLineGap=150 # Max allowed gap between line for joining them
            )
 
# Iterate over points
for points in lines:
      # Extracted points nested in the list
    x1,y1,x2,y2=points[0]
    # Draw the lines joing the points
    # On the original image
    cv2.line(image,(x1,y1),(x2,y2),(0,255,0),5)
    # Maintain a simples lookup list for points
    lines_list.append([(x1,y1),(x2,y2)])
     
# Save the result image
cv2.imwrite('detectedLines.png',image)