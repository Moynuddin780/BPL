import numpy as np
import pandas as pd

def preprocess_data():
    match = pd.read_csv("data/BPL_dataset_1.csv")
    delivery = pd.read_csv("data/BPL_deliveries_dataset_2.csv")

    total_score_df = delivery.groupby(['match_id','inning']).sum()['total_runs'].reset_index()
    total_score_df = total_score_df[total_score_df['inning']==1]
    match_df = match.merge(total_score_df[['match_id','total_runs']],left_on='id',right_on='match_id')

    teams = [
        'Khulna Tigers', 'Rangpur Riders', 'Barishal Bulls', 'Sylhet Strikers',
        'Rajshahi Kings', 'Dhaka Dominators', 'Comilla Victorians', 'Chattogram Challengers'
    ]
    match_df = match_df[match_df['team1'].isin(teams) & match_df['team2'].isin(teams)]
    match_df = match_df[['match_id','city','winner','total_runs']]
    delivery_df = match_df.merge(delivery,on='match_id')
    delivery_df = delivery_df[delivery_df['inning']==2]

    delivery_df['current_score'] = delivery_df.groupby('match_id')['total_runs_y'].cumsum()
    delivery_df['runs_left'] = delivery_df['total_runs_x'] - delivery_df['current_score']
    delivery_df['balls_left'] = 126 - (delivery_df['over']*6 + delivery_df['ball'])
    delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna("0").apply(lambda x: x if x=="0" else "1").astype(int)
    wickets = delivery_df.groupby('match_id')['player_dismissed'].cumsum().values
    delivery_df['wickets'] = 10 - wickets
    delivery_df['crr'] = (delivery_df['current_score']*6) / (120 - delivery_df['balls_left'])
    delivery_df['rrr'] = (delivery_df['runs_left']*6) / (delivery_df['balls_left'])
    delivery_df['result'] = delivery_df.apply(lambda row: 1 if row['batting_team']==row['winner'] else 0, axis=1)

    final_df = delivery_df[['batting_team','bowling_team','city','runs_left','balls_left','wickets','total_runs_x','crr','rrr','result']]
    final_df.dropna(inplace=True)
    final_df = final_df[final_df['balls_left']!=0]

    return delivery_df, final_df


def match_progression(x_df, match_id, pipe): 
    match = x_df[x_df['match_id'] == match_id]
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
    temp_df['wickets_in_over'] = (np.array(new_wickets) - np.array(wickets))[:temp_df.shape[0]]

    return temp_df[['end_of_over', 'runs_after_over', 'wickets_in_over', 'lose', 'win']], target
