
import pandas as pd
import json
import os
import sys
 
# Set encoding to utf-8 for stdout
sys.stdout.reconfigure(encoding='utf-8')

data_dir = r"c:/Users/AMG-Feven/Downloads/PL_2024_course (extract.me)"
event_dir = os.path.join(data_dir, "event_data")

output = []

def log(msg):
    output.append(str(msg))
    print(msg)

# Read matches
try:
    matches = pd.read_parquet(os.path.join(data_dir, "matches.parquet"))
    log(f"Matches Columns: {matches.columns.tolist()}")
    log(f"First match: {matches.iloc[0].to_dict()}")
except Exception as e:
    log(f"Error reading matches: {e}")

# Read players
try:
    players = pd.read_parquet(os.path.join(data_dir, "players.parquet"))
    log(f"Players Columns: {players.columns.tolist()}")
    log(f"First player: {players.iloc[0].to_dict()}")
except Exception as e:
    log(f"Error reading players: {e}")

# Read one event file
try:
    first_event_file = os.listdir(event_dir)[0]
    with open(os.path.join(event_dir, first_event_file), 'r', encoding='utf-8') as f:
        events = json.load(f)
    
    log(f"Event file: {first_event_file}")
    if isinstance(events, list):
        log(f"Number of events: {len(events)}")
        if len(events) > 0:
            log(f"First event keys: {list(events[0].keys())}")
        
        # Find a pass
        pass_event = next((e for e in events if e.get('type', {}).get('primary') == 'pass'), None)
        if pass_event:
            log("Sample Pass Event:")
            log(json.dumps(pass_event, indent=2))
        else:
            log("No pass found")
            
        # Find a shot
        shot_event = next((e for e in events if e.get('type', {}).get('primary') == 'shot'), None)
        if shot_event:
            log("Sample Shot Event:")
            log(json.dumps(shot_event, indent=2))
        else:
            types = set()
            for e in events[:500]:
                if 'type' in e and 'primary' in e['type']:
                    types.add(e['type']['primary'])
            log(f"Event types found (no shot found): {types}")
            
except Exception as e:
    log(f"Error reading event file: {e}")

with open("inspection_output.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output))
