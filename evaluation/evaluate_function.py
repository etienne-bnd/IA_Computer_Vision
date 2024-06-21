
import time 

def stable_matching(wanteds: list, preferences: dict):
    """
    Computes a stable matching using the Gale-Shapley algorithm.
    
    Parameters:
    wanteds (list): A list of items that need to be matched.
    preferences (dict): A dictionary where keys are entities needing to be matched with `wanteds`,
                        and values are lists of preferences for the corresponding entity.
    
    Returns:
    dict: A dictionary where keys are the entities and values are their matched items from `wanteds`.
    """
    result = {}
    assert len(wanteds) == len(preferences.keys()), "No matching possible due to different lengths"
    
    # Free list of wanteds
    free_wanteds = wanteds[:]
    
    # Tracking proposals made by each wanted item
    proposals = {w: [] for w in free_wanteds}
    
    # While there are free wanteds that can make proposals
    while free_wanteds:
        for w in free_wanteds[:]:  # Copy the list to iterate over as we modify the original
            # Try to propose to the most preferred entity that has not yet been proposed to
            for entity in preferences:
                if w not in proposals[entity]:
                    proposals[entity].append(w)
                    # If the entity is not yet matched, accept the proposal
                    if entity not in result:
                        result[entity] = w
                        free_wanteds.remove(w)
                        break
                    else:
                        # If already matched, check preference for current vs new proposal
                        current_match = result[entity]
                        current_index = preferences[entity].index(current_match)
                        new_index = preferences[entity].index(w)
                        
                        if new_index < current_index:
                            # New proposal is better, replace old match
                            result[entity] = w
                            free_wanteds.append(current_match)
                            free_wanteds.remove(w)
                            break
                        # If not preferred, continue to next preference
                        
    return result



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
            cx_pred,cy_pred= (pred_box[0]+(pred_box[2]//2)), (pred_box[1]+(pred_box[3]//2))
            cx_actual,cy_actual= (actual_box[0]+(actual_box[2]//2)), (actual_box[1]+(actual_box[3]//2))
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
        cx_pred,cy_pred= (pred_box[0]+(pred_box[2]//2)), (pred_box[1]+(pred_box[3]//2))
        #cx_actual,cy_actual = (actual_box[0]+(actual_box[2]//2)), (actual_box[1]+(actual_box[3]//2))
        # loss+=round(((cx_pred - cx_actual)**2 + (cy_pred - cy_actual)**2)**0.5)
        # print('x',cx_pred,cx_actual)
        # print('y',cy_pred,cy_actual)
        # print('loss',loss)
        if cx_pred<actual_box[0] or cx_pred>(actual_box[0]+actual_box[2]) or cy_pred<actual_box[1] or cy_pred>(actual_box[1]+actual_box[3]):
            loss+=1
    return(loss)
