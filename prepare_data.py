
import os
import json
import pandas as pd
import numpy as np

data_dir = os.getcwd()
event_dir = os.path.join(data_dir, "event_data")
output_file = os.path.join(data_dir, "data_tables", "passes_dataset.parquet")

if not os.path.exists(os.path.join(data_dir, "data_tables")):
    os.makedirs(os.path.join(data_dir, "data_tables"))

pass_list = []
shot_list = []

files = [f for f in os.listdir(event_dir) if f.endswith('.json')]

print(f"Processing {len(files)} files...")

for idx, filename in enumerate(files):
    filepath = os.path.join(event_dir, filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        events = data.get('events', [])
        match_id = str(data.get('matchId', filename.replace('.json', '')))
        
        for e in events:
            # Ensure minute and second are integers
            min_val = e.get('minute')
            sec_val = e.get('second')
            if min_val is None: min_val = 0
            if sec_val is None: sec_val = 0
            ts = int(min_val) * 60 + int(sec_val)

            e_type = e.get('type', {}).get('primary')
            if e_type == 'pass':
                pass_obj = e.get('pass')
                if not pass_obj: continue
                pass_info = {
                    'match_id': match_id,
                    'id': e.get('id'),
                    'player_id': e.get('player', {}).get('id'),
                    'team_id': str(e.get('team', {}).get('id')),
                    'x': e.get('location', {}).get('x'),
                    'y': e.get('location', {}).get('y'),
                    'end_x': pass_obj.get('endLocation', {}).get('x'),
                    'end_y': pass_obj.get('endLocation', {}).get('y'),
                    'timestamp': ts
                }
                pass_list.append(pass_info)
            elif e_type == 'shot':
                shot_obj = e.get('shot')
                if not shot_obj: continue
                shot_info = {
                    'match_id': match_id,
                    'team_id': str(e.get('team', {}).get('id')),
                    'timestamp': ts,
                    'xg': shot_obj.get('xg', 0)
                }
                shot_list.append(shot_info)
    except Exception as ex:
        continue

passes_df = pd.DataFrame(pass_list)
shots_df = pd.DataFrame(shot_list)

# drop duplicates or NaNs if any
passes_df.dropna(subset=['timestamp', 'match_id', 'team_id'], inplace=True)
shots_df.dropna(subset=['timestamp', 'match_id', 'team_id'], inplace=True)

# sort by timestamp
passes_df = passes_df.sort_values(by='timestamp').reset_index(drop=True)
shots_df = shots_df.sort_values(by='timestamp').reset_index(drop=True)

print(f"Merged {len(passes_df)} passes and {len(shots_df)} shots.")

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
