
import os
import json
import pandas as pd
import numpy as np

data_dir = r"c:/Users/AMG-Feven/Downloads/PL_2024_course (extract.me)"
event_dir = os.path.join(data_dir, "event_data")
output_file = os.path.join(data_dir, "passes_dataset.parquet")

def get_seconds(minute, second):
    return minute * 60 + second

pass_list = []
shot_list = []

files = [
    f for f in os.listdir(event_dir) if f.endswith('.json')]

for idx, filename in enumerate(files):
    if idx % 50 == 0:
        print(f"Processing file {idx}/{len(files)}")
    
    filepath = os.path.join(event_dir, filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        events = data.get('events', [])
        match_id = data.get('matchId', filename.replace('.json', '')) # fallback
        
        for e in events:
            e_type = e.get('type', {}).get('primary')
            
            if e_type == 'pass':
                pass_obj = e.get('pass')
                if not pass_obj: continue
               
                
                start_loc = e.get('location', {})
                end_loc = pass_obj.get('endLocation', {})
                
                pass_info = {
                    'match_id': match_id,
                    'id': e.get('id'),
                    'player_id': e.get('player', {}).get('id'),
                    'team_id': e.get('team', {}).get('id'),
                    'minute': e.get('minute'),
                    'second': e.get('second'),
                    'x': start_loc.get('x'),
                    'y': start_loc.get('y'),
                    'end_x': end_loc.get('x'),
                    'end_y': end_loc.get('y'),
                    'timestamp': get_seconds(e.get('minute', 0), e.get('second', 0))
                }
                pass_list.append(pass_info)
                
            elif e_type == 'shot':
                shot_obj = e.get('shot')
                if not shot_obj: continue
                
                shot_info = {
                    'match_id': match_id,
                    'team_id': e.get('team', {}).get('id'),
                    'timestamp': get_seconds(e.get('minute', 0), e.get('second', 0)),
                    'xg': shot_obj.get('xg', 0),
                    'is_goal': shot_obj.get('isGoal', False)
                }
                shot_list.append(shot_info)
                
    except Exception as e:
        print(f"Error processing {filename}: {e}")

print("Creating DataFrames...")
passes_df = pd.DataFrame(pass_list)
shots_df = pd.DataFrame(shot_list)

print("Passes:", len(passes_df))
print("Shots:", len(shots_df))


# Sort 
passes_df.sort_values(['match_id', 'timestamp'], inplace=True)
shots_df.sort_values(['match_id', 'timestamp'], inplace=True)



print("Merging shots to passes:")
passes_df['match_id'] = passes_df['match_id'].astype(str)
shots_df['match_id'] = shots_df['match_id'].astype(str)



shots_df = shots_df.rename(columns={'timestamp': 'shot_timestamp', 'xg': 'shot_xg', 'is_goal': 'shot_is_goal'})

shots_subset = shots_df[['match_id', 'team_id', 'shot_timestamp', 'shot_xg', 'shot_is_goal']].copy()


merged = pd.merge_asof(
    passes_df.sort_values('timestamp'),
    shots_subset.sort_values('shot_timestamp'),
    left_on='timestamp',
    right_on='shot_timestamp',
    by=['match_id', 'team_id'],
    direction='forward',
    tolerance=15
)



merged['has_shot'] = merged['shot_xg'].notna().astype(int)
merged['shot_xg'] = merged['shot_xg'].fillna(0) 

print("Saving to parquet...")
merged.to_parquet(output_file, index=False)
print("Done.")
