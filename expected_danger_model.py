
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
import os
import matplotlib.pyplot as plt
import seaborn as sns


data_dir = os.getcwd()
passes_file = os.path.join(data_dir, "data_tables", "passes_dataset.parquet")
minutes_file = os.path.join(data_dir, "data_tables", "minutes.parquet")
players_file = os.path.join(data_dir, "data_tables", "players.parquet")


print("Loading data...")
df = pd.read_parquet(passes_file)

# feature Eng.
df['start_x_sq'] = df['x']**2
df['start_y_sq'] = df['y']**2
df['end_x_sq'] = df['end_x']**2
df['end_y_sq'] = df['end_y']**2
df['dx'] = df['end_x'] - df['x']
df['dy'] = df['end_y'] - df['y']
df['dist_pass'] = np.sqrt(df['dx']**2 + df['dy']**2)
df['dist_to_goal_end'] = np.sqrt((100 - df['end_x'])**2 + (50 - df['end_y'])**2)
df['angle_to_goal_end'] = np.abs(np.arctan2(50 - df['end_y'], 100 - df['end_x']))

features = [
    'x', 'y', 'end_x', 'end_y',
    'start_x_sq', 'start_y_sq', 'end_x_sq', 'end_y_sq',
    'dist_pass', 'dist_to_goal_end', 'angle_to_goal_end'
]

# Logistic Regression: P(Shot | Pass)
df_model = df.dropna(subset=features + ['has_shot', 'shot_xg']).copy()
X = df_model[features]
y_prob = df_model['has_shot']

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X, y_prob)
df['prob_shot'] = log_reg.predict_proba(df[features])[:, 1]

# Linear Regression: Shot -> xG
shot_df = df_model[df_model['has_shot'] == 1]
lin_reg = LinearRegression()
lin_reg.fit(shot_df[features], shot_df['shot_xg'])
df['pred_xg_if_shot'] = lin_reg.predict(df[features])

# Calculate Expected Danger
df['expected_danger'] = (df['prob_shot'] * df['pred_xg_if_shot']).clip(lower=0)

# Rank Players
try:
    players = pd.read_parquet(players_file)
    mins = pd.read_parquet(minutes_file)
    
    stats = df.groupby('player_id').agg({
        'expected_danger': 'sum',
        'has_shot': 'sum',
        'id': 'count'
    }).rename(columns={'has_shot': 'danger_passes', 'id': 'total_passes'})
    
    total_minutes = mins.groupby('player_id')['minutes'].sum().reset_index()
    merged_stats = stats.reset_index().merge(total_minutes, on='player_id')
    merged_stats = merged_stats.merge(players[['player_id', 'short_name', 'role']], on='player_id')
    
    merged_stats['ed_per_90'] = (merged_stats['expected_danger'] / merged_stats['minutes']) * 90
    merged_stats['danger_passes_per_90'] = (merged_stats['danger_passes'] / merged_stats['minutes']) * 90
    
    final_stats = merged_stats[merged_stats['minutes'] > 500].copy()
    
    print("\nTop 10 Players by Expected Danger per 90:")
    print(final_stats.sort_values('ed_per_90', ascending=False)[['short_name', 'role', 'ed_per_90']].head(10).to_string(index=False))

except Exception as e:
    print(f"Error in ranking: {e}")


print("\nGenerating visualizations...")
sns.set_theme(style="whitegrid")

def draw_pitch(ax):
    plt.plot([0,100,100,0,0],[0,0,100,100,0], color="black")
    plt.plot([50,50],[0,100], color="black")
    plt.plot([0,17,17,0],[21,21,79,79], color="black")
    plt.plot([100,83,83,100],[21,21,79,79], color="black")
    ax.add_artist(plt.Circle((50,50), 9.15, color="black", fill=False))

# Heatmap
fig, ax = plt.subplots(figsize=(10, 7))
draw_pitch(ax)
high_danger = df[df['expected_danger'] > 0.05]
sns.kdeplot(x=high_danger['x'], y=high_danger['y'], fill=True, alpha=0.6, cmap='magma', ax=ax, levels=10)
plt.title("High Expected Danger Pass Origins")
plt.axis('off')
plt.savefig("danger_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()

# Scatter Plot
fig, ax = plt.subplots(figsize=(10, 8))
sns.scatterplot(data=final_stats, x='danger_passes_per_90', y='ed_per_90', alpha=0.5)
top_10 = final_stats.sort_values('ed_per_90', ascending=False).head(10)
for _, row in top_10.iterrows():
    plt.text(row['danger_passes_per_90'], row['ed_per_90'], row['short_name'], fontsize=9)
plt.title("Expected Danger vs Danger Passes per 90")
plt.savefig("quality_vs_quantity.png", dpi=300, bbox_inches='tight')
plt.close()

print("Process complete.")
