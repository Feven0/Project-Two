
import os
import json
import pandas as pd
import numpy as np

data_dir = os.getcwd()
event_dir = os.path.join(data_dir, "event_data")
output_file = os.path.join(data_dir, "passes_dataset.parquet")

pass_list = []
shot_list = []

files = [f for f in os.listdir(event_dir) if f.endswith('.json')]

for idx, filename in enumerate(files):
    filepath = os.path.join(event_dir, filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        events = data.get('events', [])
        match_id = data.get('matchId', filename.replace('.json', ''))
        
        for e in events:
            e_type = e.get('type', {}).get('primary')
            if e_type == 'pass':
                pass_obj = e.get('pass')
                if not pass_obj: continue
                pass_info = {
                    'match_id': str(match_id),
                    'id': e.get('id'),
                    'player_id': e.get('player', {}).get('id'),
                    'team_id': e.get('team', {}).get('id'),
                    'x': e.get('location', {}).get('x'),
                    'y': e.get('location', {}).get('y'),
                    'end_x': pass_obj.get('endLocation', {}).get('x'),
                    'end_y': pass_obj.get('endLocation', {}).get('y'),
                    'timestamp': e.get('minute', 0) * 60 + e.get('second', 0)
                }
                pass_list.append(pass_info)
            elif e_type == 'shot':
                shot_obj = e.get('shot')
                if not shot_obj: continue
                shot_info = {
                    'match_id': str(match_id),
                    'team_id': e.get('team', {}).get('id'),
                    'timestamp': e.get('minute', 0) * 60 + e.get('second', 0),
                    'xg': shot_obj.get('xg', 0)
                }
                shot_list.append(shot_info)
    except:
        continue

passes_df = pd.DataFrame(pass_list)
shots_df = pd.DataFrame(shot_list)

passes_df.sort_values(['match_id', 'timestamp'], inplace=True)
shots_df.sort_values(['match_id', 'timestamp'], inplace=True)

merged = pd.merge_asof(
    passes_df,
    shots_df.rename(columns={'timestamp': 'shot_ts', 'xg': 'shot_xg'}),
    left_on='timestamp',
    right_on='shot_ts',
    by=['match_id', 'team_id'],
    direction='forward',
    tolerance=15
)

merged['has_shot'] = merged['shot_xg'].notna().astype(int)
merged['shot_xg'] = merged['shot_xg'].fillna(0) 

merged.to_parquet(output_file, index=False)
print("Dataset generated.")
