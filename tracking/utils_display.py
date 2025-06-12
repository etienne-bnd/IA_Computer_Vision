import cv2 

def draw_ellipse(frame,boxes,color):
    """ Draw transparent ellipse over the original frame, representing each player """
    overlay = frame.copy()
    
    for box in boxes:
        (x, y, w, h)=box
        axes = (min(int(w),int(h)), min(int(w),int(h)))
        center = int(x + w // 2), int(y + h // 2)
        cv2.ellipse(overlay, center, axes, 0, 0, 360, color, -1)
    alpha = 0.1 + 0.3*(color[1]/255) + 0.3 *(color[0]/255) # Transparency factor
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
def additional_displays(frame,boxes,speeds,color=(255, 0, 0),offset=0):
    """ Function for displaying the bounding box, the ID and the speed of the players on the frame"""
    draw_ellipse(frame,boxes,color)
    for i,box in enumerate(boxes):
        (x, y, w, h) = box 
        cx,cy=int(x+w//2),int(y+h//2)
        try:
            cv2.putText(frame, "v: "+str(speeds[i]), (cx - 30, cy +offset + 40), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = color, thickness= 2)
            cv2.putText(frame, str(i+1), (cx,cy +offset - 40), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color =  color, thickness = 2)
        except:
            pass
