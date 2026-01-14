
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
import os
 
data_dir = r"c:/Users/AMG-Feven/Downloads/PL_2024_course (extract.me)"
passes_file = os.path.join(data_dir, "passes_dataset.parquet")
minutes_file = os.path.join(data_dir, "minutes.parquet")
players_file = os.path.join(data_dir, "players.parquet")

report_file = os.path.join(data_dir, "report.md")

def log(msg):
    print(msg)
    with open(report_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n\n")

# Reset report
with open(report_file, "w", encoding="utf-8") as f:
    f.write("# Expected Danger Model Report\n\n")

# Load Data
log("## 1. Data Loading")
df = pd.read_parquet(passes_file)
log(f"Loaded {len(df)} passes.")

# Feature Engineering
log("## 2. Feature Engineering")

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

# drop z NaNs
df_model = df.dropna(subset=features + ['has_shot', 'shot_xg']).copy()

log(f"Features used: {features}")

# Model 1: Logistic Regression (Pass -> Shot within 15s)
log("## 3. Logistic Regression: P(Shot | Pass)")

X = df_model[features]
y_prob = df_model['has_shot']



log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X, y_prob)

log(f"Model 1 Score (fraction correctly classified, biased by class imbalance): {log_reg.score(X, y_prob):.4f}")
log(f"Coefficients:\n{list(zip(features, log_reg.coef_[0]))}")

df['prob_shot'] = log_reg.predict_proba(df[features])[:, 1]


#Model 2: Linear Regression (Shot -> xG)



shot_df = df_model[df_model['has_shot'] == 1]
log(f"Training Model 2 on {len(shot_df)} passes that led to shots.")

X_shot = shot_df[features]
y_xg = shot_df['shot_xg']

lin_reg = LinearRegression()
lin_reg.fit(X_shot, y_xg)

log(f"Model 2 R2 Score: {lin_reg.score(X_shot, y_xg):.4f}")
log(f"Coefficients:\n{list(zip(features, lin_reg.coef_))}")



df['pred_xg_if_shot'] = lin_reg.predict(df[features])

# Expected Danger



df['expected_danger'] = df['prob_shot'] * df['pred_xg_if_shot']
df['expected_danger'] = df['expected_danger'].clip(lower=0)



# 6. Rank Players

try:
    players = pd.read_parquet(players_file)
    mins = pd.read_parquet(minutes_file)
    
    # Aggregate stats per player
    stats = df.groupby('player_id').agg({
        'expected_danger': 'sum',
        'has_shot': 'sum',
        'id': 'count'
    }).rename(columns={'has_shot': 'danger_passes', 'id': 'total_passes'})
    
    total_minutes = mins.groupby('player_id')['minutes'].sum().reset_index()
    
    merged_stats = stats.reset_index().merge(total_minutes, on='player_id')
    merged_stats = merged_stats.merge(players[['player_id', 'short_name', 'role']], on='player_id')
    
    # Per 90
    merged_stats['ed_per_90'] = (merged_stats['expected_danger'] / merged_stats['minutes']) * 90
    merged_stats['danger_passes_per_90'] = (merged_stats['danger_passes'] / merged_stats['minutes']) * 90
    
    final_stats = merged_stats[merged_stats['minutes'] > 500].copy()
    
    # Rank by ED per 90
    ranked_ed = final_stats.sort_values('ed_per_90', ascending=False)
    
    log("### Top 10 Players by Expected Danger per 90")
    log(ranked_ed[['short_name', 'role', 'ed_per_90', 'danger_passes_per_90', 'total_passes', 'minutes']].head(10).to_string(index=False))
    
    log("### Top 10 Players by Danger Passes per 90")
    ranked_dp = final_stats.sort_values('danger_passes_per_90', ascending=False)
    log(ranked_dp[['short_name', 'role', 'ed_per_90', 'danger_passes_per_90', 'total_passes', 'minutes']].head(10).to_string(index=False))
    
    # Correlation
    corr = final_stats['ed_per_90'].corr(final_stats['danger_passes_per_90'])
    log(f"Correlation between ED/90 and Danger Passes/90: {corr:.3f}")
    
    # Group by Position
    log("### Ranking by Position")
    for role in final_stats['role'].unique():
        log(f"#### {role}")
        role_df = final_stats[final_stats['role'] == role].sort_values('ed_per_90', ascending=False)
        log(role_df[['short_name', 'ed_per_90', 'danger_passes_per_90']].head(5).to_string(index=False))

except Exception as e:
    log(f"Error in ranking: {e}")

# ... existing code end ...

# 7. Visualizations
log("## 7. Generating Visualizations")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Arc, Rectangle, ConnectionPatch

# Style
sns.set_theme(style="whitegrid")

def draw_pitch(ax):
    # Pitch Outline & Centre Line
    plt.plot([0,0],[0,100], color="black")
    plt.plot([0,100],[100,100], color="black")
    plt.plot([100,100],[100,0], color="black")
    plt.plot([100,0],[0,0], color="black")
    plt.plot([50,50],[0,100], color="black")

    # Left Penalty Area
    plt.plot([17,17],[21,79], color="black")
    plt.plot([0,17],[79,79], color="black")
    plt.plot([17,0],[21,21], color="black")

    # Right Penalty Area
    plt.plot([83,83],[21,79], color="black")
    plt.plot([100,83],[79,79], color="black")
    plt.plot([83,100],[21,21], color="black")

    # Centre Circle
    circle = plt.Circle((50,50), 9.15, color="black", fill=False)
    ax.add_artist(circle)

# Figure 1: Heatmap of Expected Danger (Pass Origins)
# Filter for passes with non-negligible ED
high_danger_passes = df[df['expected_danger'] > 0.05]

fig, ax = plt.subplots(figsize=(10, 7))
draw_pitch(ax)
# KDE of where dangerous passes start
sns.kdeplot(
    x=high_danger_passes['x'], 
    y=high_danger_passes['y'], 
    fill=True, 
    alpha=0.6, 
    cmap='magma', 
    ax=ax,
    levels=10
)
plt.title("Origin of High Expected Danger Passes (> 0.05)", fontsize=14)
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.axis('off')
plt.savefig("danger_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()
log("Generated `danger_heatmap.png`")

# Figure 2: Scatter Plot (ED vs Danger Passes)
# Uses final_stats
try:
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot all
    sns.scatterplot(data=final_stats, x='danger_passes_per_90', y='ed_per_90', alpha=0.5, color='gray')
    
    # Highlight Top 10 ED
    top_players = final_stats.sort_values('ed_per_90', ascending=False).head(10)
    sns.scatterplot(data=top_players, x='danger_passes_per_90', y='ed_per_90', s=100, color='red')
    
    # Label them
    for _, row in top_players.iterrows():
        plt.text(
            row['danger_passes_per_90']+0.1, 
            row['ed_per_90'], 
            row['short_name'], 
            fontsize=9, 
            weight='bold'
        )
        
    plt.title("Quality vs Quantity: Expected Danger vs Danger Passes per 90", fontsize=14)
    plt.xlabel("Danger Passes per 90", fontsize=12)
    plt.ylabel("Expected Danger per 90", fontsize=12)
    plt.savefig("quality_vs_quantity.png", dpi=300, bbox_inches='tight')
    plt.close()
    log("Generated `quality_vs_quantity.png`")
    
except Exception as e:
    log(f"Error plotting scatter: {e}")

# Figure 3: Bar Chart Top 10
try:
    fig, ax = plt.subplots(figsize=(10, 6))
    top_10 = final_stats.sort_values('ed_per_90', ascending=False).head(10)
    sns.barplot(data=top_10, x='ed_per_90', y='short_name', palette='viridis')
    plt.title("Top 10 Players by Expected Danger per 90", fontsize=14)
    plt.xlabel("Expected Danger / 90")
    plt.ylabel("")
    plt.savefig("top_10_ed.png", dpi=300, bbox_inches='tight')
    plt.close()
    log("Generated `top_10_ed.png`")
except Exception as e:
    log(f"Error plotting bar chart: {e}")

print("Analysis and Visualization complete.")
