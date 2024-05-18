import cv2 

def draw_ellipse(frame,trackers):
    # Calculate center and axes lengths for the ellipse
    
    overlay = frame.copy()
    axes = (100, 100)
    for tracker in trackers.values():
        (x, y, w, h)=tracker.current_box
        center = (x + w // 2, y + h // 2)
        cv2.ellipse(overlay, center, axes, 0, 0, 360, (0, 255, 0), -1)
        alpha = 0.2  # Transparency factor
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
def additional_displays(frame,trackers):
    """ Function for displaying the bounding box, the ID and the speed of the players on the frame"""
    draw_ellipse(frame,trackers)
    for tracker in trackers.values():
        (x, y, w, h)=tracker.current_box
        cx,cy=x+w//2,y+h//2
        cv2.putText(frame, str(tracker.tracker_id), (cx,cy + 40), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.putText(frame, "v: "+str(tracker.speed), (cx - 20, cy - 40), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0, 0), 3)

