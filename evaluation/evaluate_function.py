
import math
from evaluation import utils_eval

def compute_mapping(predicted_boxes,actual_boxes):
    """ conputes the mapping: player_ids used in the predictions -> actual player_ids """
    mapping={}
    for player_id,pred_box in enumerate(predicted_boxes):
        best_actual_player_id,min_distance=None,math.inf
        for actual_player_id,actual_box in actual_boxes.items():    
            cx_pred,cy_pred= (pred_box[0]-pred_box[2]//2), (pred_box[1]-pred_box[3]//2)
            cx_actual,cy_actual= (actual_box[0]-actual_box[2]//2), (actual_box[1]-actual_box[3]//2)
            d=(cx_pred - cx_actual)**2 + (cy_pred - cy_actual)**2
            if d<min_distance:
                best_actual_player_id,min_distance=actual_player_id,d
        mapping[player_id]=best_actual_player_id
    return(mapping)

def evaluate(mapping,predicted_boxes,actual_boxes):
    """ mapping: player_ids used in the predictions -> actual player_ids """
    loss=0
    for player_id,pred_box in enumerate(predicted_boxes):
        actual_player_id=mapping[player_id]
        actual_box=actual_boxes[actual_player_id]
        cx_pred,cy_pred= (pred_box[0]-pred_box[2]//2), (pred_box[1]-pred_box[3]//2)
        cx_actual,cy_actual= (actual_box[0]-actual_box[2]//2), (actual_box[1]-actual_box[3]//2)
        loss+=(cx_pred - cx_actual)**2 + (cy_pred - cy_actual)**2
    return(loss)
