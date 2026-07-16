import pickle

import numpy as np
import pandas as pd
import streamlit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle as pkl

df = pd.read_csv("Teen_Mental_Health_Dataset.csv")
print(df['addiction_level'].unique().min())
#print(df['platform_usage'].unique())
#print(df.columns.unique().tolist())
#'age', 'gender', 'daily_social_media_hours', 'platform_usage', 'sleep_hours', 'screen_time_before_sleep',
# 'academic_performance', 'physical_activity', 'social_interaction_level', 'stress_level', 'anxiety_level',
# 'addiction_level', 'depression_label']
#[]
#Data divided into X and Y category
x = df[['age', 'gender', 'daily_social_media_hours', 'platform_usage', 'sleep_hours',
'screen_time_before_sleep', 'academic_performance', 'physical_activity',
'social_interaction_level', 'stress_level', 'anxiety_level', 'addiction_level']]
y = df['depression_label']
#print(df['depression_label'].describe())
#print(len(df['depression_label']))
#print(df['depression_label'].value_counts())

for col in x.select_dtypes(include=['object', 'string']).columns:
    le = LabelEncoder()
    x[col] = le.fit_transform(x[col].astype(str)).astype(int)

# Data Splitting into x_train, x_test, y_train, y_test categories
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LogisticRegression(solver='liblinear', max_iter=500)
model.fit(x_train, y_train)
#print(model.score(x_test, y_test))
#print(model.predict(x_test))
#print(df.isnull().sum())

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

