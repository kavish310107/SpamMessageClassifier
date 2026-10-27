from DataPreprocessing import transform_text
import pickle
from pathlib import Path

model = pickle.load(open(Path(__file__).with_name('model.pkl'), 'rb'))
vectorizer = pickle.load(open(Path(__file__).with_name('vectorizer.pkl'), 'rb'))

#load the preprocessor,  text-vectorizer and saved model

def classify_SMS(text) :
    
    #preprocess the new ip data, i.e tolowercase, stemming, tokenizaion,etc
    data = transform_text(text)
    
    #transform the data based on already fitted vectorizer and predict outcome based on already trained model
    vectored_data = vectorizer.transform([' '.join(data)])
    prediction  = model.predict(vectored_data)
    
    return prediction[0]


#THIS IS THE FUNCTION WHICH WILL BE CALLED BY APP.PY, AND THIS IS HOW THE ENTIRE ML MODEL TRAINING TO DEPLOYMENT WORKS..