import cv2

def test_oob(box,bounds):
    """Choose whether or not to keep a box, depending on  if it is within the bounds of the field[(xmin,xmax),(ymin,ymax)]"""
    x,y,w,h=box
    xmin,xmax=bounds[0]
    ymin,ymax=bounds[1]
    tol=0.01
    return((h>2) and ((1+tol)*(x+h//2)>= xmin) and ((1-tol)*(x+h//2)<=xmax) and ((1+tol)*(y+w//2)>= ymin) and ((1-tol)*(y+w//2)<=ymax))

def init_bounds(width,height,scale=0.9):
  bounds = [((1-scale)*width,scale*width),((1-scale)*height,scale*height)]
  return(bounds)


def input_bounds(frame):
    width,height,_=frame.shape
    cv2.namedWindow("first frame", cv2.WINDOW_NORMAL) 
    cv2.resizeWindow("first frame", 1000, 600) 
    x, y, w, h = cv2.selectROI('first frame', frame, showCrosshair=False)
    cv2.destroyWindow('first frame')
    return([(x, x+w),(y,y+h)])

def reshape_box(box):
    """ Downscale to 75 percentand make it  a square """
    x,y,w,h=box
    w_new=(3*w)//4
    h_new=(3*h)//4
    w_new,h_new=min(w_new,h_new),min(w_new,h_new)
    x_new=x+w//2 - w_new//2
    y_new=y+h//2 - h_new//2
    return(x_new,y_new,w_new,h_new)

def sort_players(boxes):
    return(sorted(boxes,key=(lambda box: box[0]+100*box[1])))