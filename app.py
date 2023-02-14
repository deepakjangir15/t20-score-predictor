import pickle
import streamlit as st
import pandas as pd
import numpy as np

pipeline = pickle.load(open('cricket_pipe_XGB.pkl','rb'))

top_teams = ['Pakistan',
    'South Africa',
    'India',
    'New Zealand',
    'Sri Lanka',
    'West Indies',
    'Australia',
    'England',
    'Afghanistan',
    'Bangladesh',
    'Ireland',
    'Zimbabwe'] 

cites = ['Colombo',
 'Dubai',
 'Mirpur',
 'Johannesburg',
 'Auckland',
 'Harare',
 'Cape Town',
 'London',
 'Abu Dhabi',
 'Pallekele',
 'Barbados',
 'Durban',
 'Sydney',
 'Melbourne',
 'Chittagong',
 'St Lucia',
 'Wellington',
 'Nottingham',
 'Hamilton',
 'Lauderhill',
 'Centurion',
 'Manchester',
 'Sharjah',
 'Lahore',
 'Mumbai',
 'Dhaka',
 'Nagpur',
 'Southampton',
 'Mount Maunganui',
 'Kolkata',
 'Hambantota',
 'Dublin',
 'Greater Noida',
 'Delhi',
 'Trinidad',
 'Sylhet',
 'Cardiff',
 'Guyana',
 'Chandigarh',
 'Adelaide',
 'Bangalore',
 'St Kitts',
 'Christchurch',
 'Brisbane',
 'Birmingham']

st.title('T20 Score Predictor - by Deepak')

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Batting Team',sorted(top_teams))

with col2:
    bowling_team = st.selectbox('Bowling Team',sorted(top_teams))

city = st.selectbox('City', sorted(cites))

col3, col4, col5 = st.columns(3)

with col3:
    current_score = st.number_input('Current Score',step=1)

with col4:
    overs = st.number_input('Over Completed',min_value=5,max_value=20,step=1)

with col5:
    wickets = st.number_input('Wickets?',min_value=1,max_value=10,step=1)

last_five = st.number_input('Runs scored in last 5 overs',step=1)

if st.button('Predict score'):
    balls_remaining = 120 - (overs * 6)
    wickets_remaining = 10 - wickets
    crr = current_score / overs

    input_df = pd.DataFrame({'Batting_team': [batting_team], 
                             'Bowling_team': [bowling_team],
                             'city':city, 
                             'current_score': [current_score],
                             'balls_left': [balls_remaining], 
                             'wickets_left': [wickets_remaining], 
                             'crr': [crr], 
                             'last_five': [last_five]})
    
    result = pipeline.predict(input_df)

    st.text('Predicted score is ' + str(int(result[0])))
   


