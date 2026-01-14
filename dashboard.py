
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LogisticRegression, LinearRegression


st.set_page_config(page_title="Expected Danger Dashboard", layout="wide")

st.title("Premier League: Expected Danger Dashboard")

@st.cache_data
def load_data():
    data_dir = os.getcwd()
    passes_file = os.path.join(data_dir, "data_tables", "passes_dataset.parquet")
    players_file = os.path.join(data_dir, "data_tables", "players.parquet")
    minutes_file = os.path.join(data_dir, "data_tables", "minutes.parquet")
    
    df = pd.read_parquet(passes_file)
    players = pd.read_parquet(players_file)
    mins = pd.read_parquet(minutes_file)
    
 
    df['start_x_sq'] = df['x']**2
    df['start_y_sq'] = df['y']**2
    df['end_x_sq'] = df['end_x']**2
    df['end_y_sq'] = df['end_y']**2
    df['dx'] = df['end_x'] - df['x']
    df['dy'] = df['end_y'] - df['y']
    df['dist_pass'] = np.sqrt(df['dx']**2 + df['dy']**2)
    df['dist_to_goal_end'] = np.sqrt((100 - df['end_x'])**2 + (50 - df['end_y'])**2)
    df['angle_to_goal_end'] = np.abs(np.arctan2(50 - df['end_y'], 100 - df['end_x']))
    
    features = ['x', 'y', 'end_x', 'end_y', 'start_x_sq', 'start_y_sq', 'end_x_sq', 'end_y_sq', 'dist_pass', 'dist_to_goal_end', 'angle_to_goal_end']
    df_model = df.dropna(subset=features + ['has_shot', 'shot_xg']).copy()
    
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(df_model[features], df_model['has_shot'])
    df['prob_shot'] = log_reg.predict_proba(df[features])[:, 1]
    
    shot_df = df_model[df_model['has_shot'] == 1]
    lin_reg = LinearRegression()
    lin_reg.fit(shot_df[features], shot_df['shot_xg'])
    df['pred_xg'] = lin_reg.predict(df[features])
    df['expected_danger'] = (df['prob_shot'] * df['pred_xg']).clip(lower=0)
    
    stats = df.groupby('player_id').agg({'expected_danger': 'sum', 'has_shot': 'sum', 'id': 'count'}).rename(columns={'has_shot': 'danger_passes', 'id': 'total_passes'})
    total_mins = mins.groupby('player_id')['minutes'].sum().reset_index()
    merged = stats.reset_index().merge(total_mins, on='player_id').merge(players[['player_id', 'short_name', 'role']], on='player_id')
    merged['ed_per_90'] = (merged['expected_danger'] / merged['minutes']) * 90
    merged['dp_per_90'] = (merged['danger_passes'] / merged['minutes']) * 90
    
    return df, merged

def draw_pitch(ax):
    plt.plot([0,100,100,0,0],[0,0,100,100,0], color="black")
    plt.plot([50,50],[0,100], color="black")
    plt.plot([0,17,17,0],[21,21,79,79], color="black")
    plt.plot([100,83,83,100],[21,21,79,79], color="black")
    ax.add_artist(plt.Circle((50,50), 9.15, color="black", fill=False))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')

df, stats = load_data()


st.sidebar.header("Filters")
min_mins = st.sidebar.slider("Minimum Minutes", 0, 3000, 500)
selected_role = st.sidebar.multiselect("Positions", options=stats['role'].unique(), default=stats['role'].unique())
filtered_stats = stats[(stats['minutes'] >= min_mins) & (stats['role'].isin(selected_role))]

tab1, tab2, tab3 = st.tabs(["Player Rankings", "Heatmaps", "Quality vs Quantity"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Highest ED / 90")
        st.dataframe(filtered_stats.sort_values('ed_per_90', ascending=False)[['short_name', 'role', 'ed_per_90']].head(10))
    with col2:
        st.subheader("Most Danger Passes / 90")
        st.dataframe(filtered_stats.sort_values('dp_per_90', ascending=False)[['short_name', 'role', 'dp_per_90']].head(10))

with tab2:
    player_list = filtered_stats.sort_values('short_name')['short_name'].tolist()
    target_player = st.selectbox("Select Player", options=player_list)
    player_id = stats[stats['short_name'] == target_player]['player_id'].values[0]
    player_passes = df[(df['player_id'] == player_id) & (df['expected_danger'] > 0.02)]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    draw_pitch(ax)
    if len(player_passes) > 5:
        sns.kdeplot(x=player_passes['x'], y=player_passes['y'], fill=True, alpha=0.6, cmap='magma', ax=ax, levels=10)
        st.pyplot(fig)
    else:
        st.warning("Insufficient data for heatmap.")

with tab3:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=filtered_stats, x='dp_per_90', y='ed_per_90', hue='role')
    st.pyplot(fig)
