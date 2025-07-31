import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from utils import preprocess_data, match_progression

# Load Model
with open("model_pipeline.pkl", "rb") as f:
    pipe = pickle.load(f)

# Load Data
delivery_df, final_df = preprocess_data()

# Streamlit UI
st.set_page_config(page_title="BPL Match Predictor", layout="wide")
st.title("🏏 BPL Match Progression & Win Predictor")

match_ids = delivery_df['match_id'].unique()
match_id = st.selectbox("Select Match ID", match_ids)

if st.button("Show Match Prediction"):
    temp_df, target = match_progression(delivery_df, match_id, pipe)
    
    st.subheader(f"🎯 Target: {target} runs")
    
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(temp_df['end_of_over'], temp_df['win'], label='Win %', color='green', linewidth=3)
    ax.plot(temp_df['end_of_over'], temp_df['lose'], label='Lose %', color='red', linewidth=3)
    ax.set_xlabel("Overs")
    ax.set_ylabel("Probability")
    ax.legend()
    st.pyplot(fig)

    st.bar_chart(temp_df.set_index("end_of_over")["runs_after_over"])

    st.subheader("📋 Match Progression Table")
    st.dataframe(temp_df)
