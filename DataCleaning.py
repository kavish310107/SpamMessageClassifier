import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(r'7. SpamDetectionClassifier\spam.csv',encoding='latin-1')

#DATA CLEANING
# EDA
#TEXT PREPROCESSING
#MODEL CREATION
#MODEL EVALUATION
#WEBSITE CREATION
#DEPLOYMENT

#1. DATA CLEANING
# 0 => not spam , 1=> spam
df = df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'])
df = df.rename(columns={'v1':'target','v2' : 'text'})
enc = LabelEncoder()
df['target'] = enc.fit_transform(df['target'])
df = df.drop_duplicates(keep='first')
# print(df['target'].value_counts())

df.to_csv(r'7. SpamDetectionClassifier\transformed_spam.csv',index=False)
