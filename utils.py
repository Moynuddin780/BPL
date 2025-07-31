# utils.py

import pandas as pd

def load_match_data():
    return pd.read_csv("data/BPL_dataset_1.csv")

def load_delivery_data():
    return pd.read_csv("data/BPL_deliveries_dataset_2.csv")

def get_unique_teams(match_df):
    teams = pd.concat([match_df['team1'], match_df['team2']]).unique()
    return sorted(teams)

def prepare_input_dataframe(batting_team, bowling_team, venue, toss_decision, toss_winner, match_df):
    input_data = {
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'venue': [venue],
        'toss_decision': [toss_decision],
        'toss_winner': [toss_winner]
    }

    # Add other required features if your model needs them
    df = pd.DataFrame(input_data)
    return df
