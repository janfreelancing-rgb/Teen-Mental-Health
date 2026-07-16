import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder
import streamlit as st
from streamlit import dataframe
import pickle


#Load Logistic Model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

#'age', 'gender', 'daily_social_media_hours', 'platform_usage', 'sleep_hours',
#'screen_time_before_sleep', 'academic_performance', 'physical_activity',
#'social_interaction_level', 'stress_level', 'anxiety_level', 'addiction_level
#st.set_page_config(page_title="Logistic Regression", layout="wide")
st.title("Teen Mental Health Prediction Application")
st.subheader("This App Predict While a Teen is Currently Suffered in Depression or Not, on The "
          "Basis of The Usage of Social Media and Other Activities in His Daily Life")
st.badge('Follow the instruction below and fill the fields')
gender = st.radio('Select Gender', ('Male', 'Female'))
gender =  0 if gender == 'Male' else 1
age = st.slider('Age', 10, 20, 12,1)
daily_social_media_hours = st.number_input('Daily Social Media Hours', 0, 10,1)
platform_usage = st.selectbox('Select Platform Usage', ('TikTok', 'Instagram','Both'))
platform_mapping = { 'TikTok' : 0, 'Instagram' : 1, 'Both' : 2 }
platform_usage = platform_mapping[platform_usage]
sleep_hours = st.number_input('Sleep Hours', 0, 10, 6,1)
screen_time_before_sleep = st.number_input('Screen Time Before Sleep', 0.0, 3.0, 1.0,0.5)
academic_performance =st.number_input('Academic Performance (Enter Value between 2--4)', 2.0, 4.0, 2.0,0.01)
physical_activity = st.number_input('Physical Activity (Enter Value between 0--2)', 0.0, 2.0, 1.0,0.1)
social_interaction_level = st.selectbox('Social Interaction Level', ('Low','Medium','High'))
social_interaction_level_mapping = {'Low':0, 'Medium':1, 'High':2}
social_interaction_level = social_interaction_level_mapping[social_interaction_level]
stress_level = st.number_input('Stress Level (Enter Value between 1--10)', 1, 10, 1,1)
anxiety_level = st.number_input('Anxiety Level (Enter Value between 1--10)', 1, 10, 1,1)
addiction_level = st.number_input('Addiction Level (Enter Value between 1--10)', 1, 10, 1,1)


# {[]}
# Creating a pipline for the input values
input_values = pd.DataFrame ({'age':[age],'gender':[gender], 'daily_social_media_hours':[daily_social_media_hours],
                              'platform_usage':[platform_usage], 'sleep_hours':[sleep_hours],
                              'screen_time_before_sleep':[screen_time_before_sleep], 'academic_performance':[academic_performance],
                             'physical_activity':[physical_activity],'social_interaction_level':[social_interaction_level],
                            'stress_level':[stress_level], 'anxiety_level':[anxiety_level], 'addiction_level': [addiction_level]
                            })
#st.write(input_values.dtypes)
#st.subheader("Input Values")
#st.dataframe(input_values)

if st.button("Predict"):
    if model.predict(input_values)==0:
        st.success("This Teen Child Mental Health is Fine, Plz Prevent him/her from Social media")
    else:
        st.warning("This Teen Child Suffered in Depression, Plz Prevent Social Media for Three Months")

