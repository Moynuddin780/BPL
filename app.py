import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

#st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache_data
def load_data():
    match = pd.read_csv("data/BPL_dataset_1.csv")
    delivery = pd.read_csv("data/BPL_deliveries_dataset_2.csv")
    return match, delivery

@st.cache_resource
def load_model():
    pipe = joblib.load("model_pipeline.pkl")
    return pipe

def match_progression(delivery_df, match_id, pipe):
    match = delivery_df[delivery_df['match_id'] == match_id]
    match = match[(match['ball'] == 6)]
    temp_df = match[['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left', 'wickets', 'total_runs_x', 'crr', 'rrr']]
    temp_df = temp_df[temp_df['balls_left'] != 0]

    result = pipe.predict_proba(temp_df)
    temp_df['lose'] = np.round(result.T[0] * 100, 1)
    temp_df['win'] = np.round(result.T[1] * 100, 1)
    temp_df['end_of_over'] = range(1, temp_df.shape[0] + 1)

    target = temp_df['total_runs_x'].values[0]
    runs = list(temp_df['runs_left'].values)
    new_runs = runs[:]
    runs.insert(0, target)
    temp_df['runs_after_over'] = np.array(runs)[:-1] - np.array(new_runs)

    wickets = list(temp_df['wickets'].values)
    new_wickets = wickets[:]  
    new_wickets.insert(0, 10)  

    wickets.append(0)  
    wickets = np.array(wickets) 
    nw = np.array(new_wickets)   
    temp_df['wickets_in_over'] = (nw - wickets)[:temp_df.shape[0]]

    temp_df = temp_df[['end_of_over', 'runs_after_over', 'wickets_in_over', 'lose', 'win']]
    return temp_df, target

def main():
    st.title("Bangladesh Premier League Win Predictor")

    match, delivery = load_data()
    pipe = load_model()

    match_ids = match['id'].unique()
    selected_match_id = st.selectbox("Select Match ID:", match_ids)

    temp_df, target = match_progression(delivery, selected_match_id, pipe)

    st.subheader(f"Match Progression (Match ID: {selected_match_id})")
    st.write(f"Target: {target} runs")

    plt.figure(figsize=(12,6))
    plt.plot(temp_df['end_of_over'], temp_df['wickets_in_over'], color='orange', linewidth=3, label='Wickets in Over')
    plt.plot(temp_df['end_of_over'], temp_df['win'], color='green', linewidth=3, label='Winning Probability (%)')
    plt.plot(temp_df['end_of_over'], temp_df['lose'], color='red', linewidth=3, label='Losing Probability (%)')
    plt.bar(temp_df['end_of_over'], temp_df['runs_after_over'], alpha=0.3, label='Runs scored in Over')
    plt.xlabel("Overs")
    plt.ylabel("Count / Probability")
    plt.legend()
    plt.title(f"Match Progression Chart for Match ID {selected_match_id}")
    st.pyplot()

if __name__ == "__main__":
    main()
