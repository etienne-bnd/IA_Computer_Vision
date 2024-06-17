
import time 

def stable_matching(wanteds: list, preferences: dict):
    """
    computes a stable matching using the following format: 
    wanteds = [34, 76, 59, 76,22, 27] 
    preferences = {'A': [34, 76, 59], 'B': [22, 27], 'C': [27, 22]}
    result = stable_matching(wanteds, preferences)

    """
    result = {}
    for preference_name, preference_numbers in preferences.items():
        for preference_number in preference_numbers:
            if preference_number in wanteds:
                wanteds.remove(preference_number)
                result[preference_name] = (preference_number, preference_numbers.index(preference_number))
                break
        
            current_preference_name_preference_number_index = preference_numbers.index(preference_number)
            for old_result_name, old_result_made in result.items():
                if old_result_made[0] == preference_number and old_result_made[1] > current_preference_name_preference_number_index:
                    result[preference_name] = (preference_number, preference_numbers.index(preference_number))
                    result.pop(old_result_name)
                    break

    for preference_name, preference_numbers in preferences.items():
        if preference_name not in result:
            result[preference_name] = -1
        else:
            result[preference_name] = result[preference_name][0]

    return dict(sorted(result.items()))



def compute_mapping(predicted_boxes,actual_boxes):
    """ 
    computes the mapping: player_ids used in the predictions -> actual player_ids 
    by using stable matching, with actual boxes in  the "wanteds" role
    """

    wanteds=list(range(len(actual_boxes)))
    preferences={}

    for player_id,pred_box in enumerate(predicted_boxes):
        l=[]
        for actual_player_id,actual_box in actual_boxes.items():    
            cx_pred,cy_pred= (pred_box[0]-(pred_box[2]//2)), (pred_box[1]-(pred_box[3]//2))
            cx_actual,cy_actual= (actual_box[0]-(actual_box[2]//2)), (actual_box[1]-(actual_box[3]//2))
            d = (cx_pred - cx_actual)**2 + (cy_pred - cy_actual)**2
            l.append((actual_player_id,d))

        preferences[player_id] = [actual_player_id for (actual_player_id,_) in sorted(l,key=lambda u: u[1])]

    mapping  = stable_matching(wanteds, preferences)
    
    return(mapping)

def evaluate(mapping,predicted_boxes,actual_boxes):
    """ mapping: player_ids used in the predictions -> actual player_ids """
    loss=0
    
    for player_id,pred_box in enumerate(predicted_boxes):
        
        actual_player_id=mapping[player_id]
        actual_box=actual_boxes[actual_player_id]

        cx_actual,cy_actual= (actual_box[0]+(actual_box[2]//2)), (actual_box[1]+(actual_box[3]//2))
        # cx_pred,cy_pred= (pred_box[0]+(pred_box[2]//2)), (pred_box[1]+(pred_box[3]//2))
        # loss+= ((cx_pred - cx_actual)**2 + (cy_pred - cy_actual)**2)**0.5
        # print('x',cx_pred,cx_actual)
        # print('y',cy_pred,cy_actual)
        # print('loss',loss)
        if cx_actual<actual_box[0] or cx_actual>(actual_box[0]+actual_box[2]//2) or cy_actual<actual_box[1] or cy_actual>(actual_box[1]+actual_box[3]):
            loss+=1
    return(loss)
