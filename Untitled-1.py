"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)
# Data dependencies
import pandas as pd
from PIL import Image

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

	# Creates a main title and subheader on your page -
	# these are static across all pages
st.title("DataLink Tweet Classifer APP")
st.subheader("Climate change tweet classification")
st.markdown("This application is a streamlit dashboard to analyze and classify predictions of Tweets sentiments")
# Creating sidebar with selection box -
# you can create multiple pages this way
st.sidebar.title("Navigation Menu")

# The main function where we will build the actual app




	

	
# Building out the "Home Information" page
options = ["Home", "Data Analysis"]
st.sidebar.selectbox("Select  Dashboard",options)

if options == 'Home':
	st.subheader('Tweets Classifier App')
	image = Image.open('resources/Logo.png')
	st.image(image, caption='TEAM JS6', use_column_width=True)



# Building out the "Home Information" page
models = ["Logistic Regression", "Random Forest Classifier", "MultinomialNB", "KNeighbors Classifier", "Linear SVC"]
st.sidebar.selectbox("Prediction Models ",models)

# Building out the "Home Information" page
options2 = ['Contact Us']
st.sidebar.selectbox("For More Info",options2)

	    ##contact page
if options2 == 'App Developers':
		st.info('Contact details in case you have any query : 🧑🏻‍🤝‍🧑🏽')
		st.write('Mbali')
		st.write('Velly')
		st.write('Andy')
		st.write('Amantle')
		st.write('Mengezi')

       # Footer











		

	


