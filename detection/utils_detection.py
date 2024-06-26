import cv2

def test_oob(box,bounds):
    """
    Determines whether a box should be kept based on its position within specified bounds.

    Args:
        box (tuple): A tuple containing the coordinates and dimensions of the box in the format (x, y, w, h).
        bounds (tuple): A tuple containing the minimum and maximum bounds for the field in the format ((xmin, xmax), (ymin, ymax)).

    Returns:
        bool: True if the box is within the bounds, otherwise False.

    The function uses a tolerance value to ensure that the center of the box (adjusted by width and height) is within the specified bounds.
    """
    x,y,w,h=box
    xmin,xmax=bounds[0]
    ymin,ymax=bounds[1]
    tol=0.01
    return((h>2) and ((1+tol)*(x+h//2)>= xmin) and ((1-tol)*(x+h//2)<=xmax) and ((1+tol)*(y+w//2)>= ymin) and ((1-tol)*(y+w//2)<=ymax) and h<150 and w<150)

def input_bounds(frame):
    """
    Allows the user to select a region of interest (ROI) within a given frame interactively.

    Args:
        frame (numpy.ndarray): The input image frame from which the ROI is to be selected.

    Returns:
        list: A list containing the bounds of the selected ROI in the format [(xmin, xmax), (ymin, ymax)].

    This function displays the input frame with an instructional text, in a resizable window and lets the user 
    interactively select a rectangular ROI. The selected ROI bounds are returned as a list of tuples representing 
    the coordinates of the top-left and bottom-right corners.
    """
    # Define the text and its properties
    text = "Draw the rectangle containing only the players"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 255, 0)  # Green color
    thickness = 2
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Calculate the text position to be centered
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = 50  # Fixed height position near the top
    
    # Draw the text on the frame
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)

    # Display the frame in a resizable window
    cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
    cv2.imshow("Select ROI", frame)
    
    # Allow the user to select the ROI
    x, y, w, h = cv2.selectROI('Select ROI', frame, showCrosshair=False)
    
    # Close the window after ROI selection
    cv2.destroyWindow('Select ROI')
    
    return [(x, x + w), (y, y + h)]

def reshape_box(box):
    """
    Downsizes the input box to 75 percent and transforms it into a square.

    Args:
        box (tuple): A tuple containing the coordinates and dimensions of the original box in the format (x, y, w, h).

    Returns:
        tuple: A tuple representing the modified box with adjusted dimensions, now a square, in the format (x_new, y_new, w_new, h_new).

    This function first reduces the width and height of the input box to 75% of their original values.
    Then, it ensures that the box becomes a square by setting its dimensions to be equal.
    Finally, it adjusts the coordinates to center the new square box within the original box.
    """
    x,y,w,h=box
    w_new=(3*w)//4
    h_new=(3*h)//4
    w_new,h_new=min(w_new,h_new),min(w_new,h_new)
    x_new=x+w//2 - w_new//2
    y_new=y+h//2 - h_new//2
    return(x_new,y_new,w_new,h_new)

def sort_players(boxes):
    """
    Sorts a list of player bounding boxes based on their positions.

    Args:
        boxes (list): A list of bounding boxes, where each box is represented as a tuple (x, y, w, h).

    Returns:
        list: A sorted list of bounding boxes based on the sum of x-coordinate and 100 times the y-coordinate.

    This function sorts the input list of bounding boxes based on the sum of each box's x-coordinate and 100 times its y-coordinate.
    """
    return(sorted(boxes,key=(lambda box: box[0]+100*box[1])))