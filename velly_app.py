#streamlit dependencies
import streamlit as st
import joblib, os

## data dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import re
import nltk

from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nlppreprocess import NLP # pip install nlppreprocess
from nltk import pos_tag
import base64
import seaborn as sns
import io

from collections import Counter
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords, wordnet  
from sklearn.feature_extraction.text import CountVectorizer   
from sklearn.feature_extraction.text import TfidfTransformer 



import warnings
warnings.filterwarnings('ignore')

from nlppreprocess import NLP
nlp = NLP()

import matplotlib.style as style 
sns.set(font_scale=1.5)
style.use('seaborn-pastel')
style.use('seaborn-poster')
from PIL import Image
from wordcloud import WordCloud

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Streamlit dependencies
import streamlit as st
import joblib,os
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)
# Data dependencies
import pandas as pd
from PIL import Image
 
# Load your raw data
raw = pd.read_csv("resources/train.csv")

vectorizer = open('resources/tfidfvect.pkl','rb')   ##  will be replaced by the cleaning and preprocessing function
tweet_cv = joblib.load(vectorizer)

st.sidebar.title('Options')

option = st.sidebar.selectbox('Which Dashboard?', ('Home','Predictions', 'Charts', 'Raw Tweets','Contact Us'))

st.header(option)

if option == 'Predictions':
    st.subheader('Climate Change Tweet Classifier')


    st.info('Make Predictions of your Tweet(s) using our ML Model')

    data_source = ['Select option', 'Single text'] ## differentiating between a single text and a dataset inpit

    source_selection = st.selectbox('What to classify?', data_source)

    # Load Our Models
    def load_prediction_models(model_file):
        #predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
        loaded_models = joblib.load(open(os.path.join(model_file),"rb"))
        return loaded_models

    # Getting the predictions
    def get_keys(val,my_dict):
        for key,value in my_dict.items():
            if val == value:
                return key

    def use_model(model,tweet_text ):
        vect_text = tweet_cv.transform([tweet_text]).toarray()
        predictor = joblib.load(open(os.path.join("resources/" + model),"rb"))
        prediction = predictor.predict(list(vect_text))
        return prediction

    if source_selection == 'Single text':
        ### SINGLE TWEET CLASSIFICATION ###
        st.subheader('Single tweet classification')

        tweet_text = st.text_area('Enter Text (max. 140 characters):') ##user entering a single text to classify and predict
        all_ml_models = ["Logistic Regression", "Random Forest Classifier", "MultinomialNB", "KNeighbors Classifier", "Linear SVC"]
        model_choice = st.selectbox("Choose Machine Learning Model",all_ml_models)
        st.info("Make a classification using our {} model".format(model_choice))

        prediction_labels = {'Negative':-1,'Neutral':0,'Positive':1,'News':2}


        if st.button('Classify'):

            st.text("Original test :\n{}".format(tweet_text))

            if model_choice == 'Logistic Regression':
                prediction = use_model("resources/lr.pkl",tweet_text )
            elif model_choice == 'Random Forest Classifier':
                prediction = use_model("resources/rfc.pkl",tweet_text )
            elif model_choice == 'MultinomialNB':
                prediction = use_model("resources/mnb.pkl",tweet_text )
            elif model_choice == 'KNeighbors Classifier':
                prediction = use_model("resources/knn.pkl",tweet_text )
            elif model_choice == 'Linear SVC':
                prediction = use_model("resources/svc.pkl",tweet_text )        
            

             
            final_result = get_keys(prediction,prediction_labels)
            st.success("Tweets Categorized as: {}".format(final_result))


if option == 'Home':
    st.subheader('Tweets Classifier App')
    from PIL import Image
    image = Image.open('resources/Logo.png')
    st.image(image, caption='Which Tweet are you?', use_column_width=True)


