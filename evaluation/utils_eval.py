
import pandas as pd

def load_annotations_from_csv(path):
    df_boxes=pd.read_csv(path,skiprows=2)
    df_player_ids=pd.read_csv(path,nrows=1)
    return(df_boxes,df_player_ids)

def from_df_to_boxes(df_boxes,df_player_ids):
    n_frames=len(df_boxes)-1
    total_players=(len(df_player_ids.columns)-5)//(4)
    summary=[]
    for frame_id in range(n_frames):
      dict_boxes={}
      for player_id in range(total_players):
        suffix=('' if player_id ==0 else '.'+str(player_id) )
        x,y,w,h=df_boxes['bb_left'+suffix].iloc[frame_id+1],df_boxes['bb_top'+suffix].iloc[frame_id+1],df_boxes['bb_width'+suffix].iloc[frame_id+1]+1,df_boxes['bb_height'+suffix].iloc[frame_id+1]
        dict_boxes[player_id]=(x,y,w,h)
      summary.append(dict_boxes)
    return(summary)
  

