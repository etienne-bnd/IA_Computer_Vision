
import pandas as pd

def load_annotations_from_csv(path):
    try:
        df_boxes=pd.read_csv(path,skiprows=2)
        df_player_ids=pd.read_csv(path,nrows=1)
        return(df_boxes,df_player_ids)
    except:
        print( "Wrong path_to_annotation name or invalid data type")
        raise ValueError
def from_df_to_boxes(df_boxes,df_player_ids,original_img_shape,desired_img_shape):
    
    (original_width,original_height),(desired_width, desired_height) = original_img_shape,desired_img_shape
    rx=desired_width/original_width
    ry=desired_height/original_height

    n_frames=len(df_boxes)-1
    total_players=(len(df_player_ids.columns)-5)//(4)
    summary=[]
    values = df_boxes.values[1:,1:] 

    for frame_id in range(n_frames):
      dict_boxes={}
      for player_id in range(total_players):
        try:
          x = values[frame_id][4*player_id+1]
          y = values[frame_id][4*player_id+2]
          h = values[frame_id][4*player_id]
          w = values[frame_id][4*player_id+3]
          dict_boxes[player_id]=(int(rx*x),int(ry*y),int(rx*w),int(ry*h))
        except:
          dict_boxes[player_id]=summary[-1][player_id]
      summary.append(dict_boxes)
    return(summary)
  

