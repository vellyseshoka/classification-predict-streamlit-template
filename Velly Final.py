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
from streamlit_option_menu import option_menu


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

with st.sidebar:
		selected = option_menu(
			menu_title = 'Navigation Menu',
			menu_icon="list", 
			options = ['Home', 'Data Analysis', 'Models', 'Contact Us'],
			icons = [ "house","info-circle", 'bar-chart-line', "envelope"],
			)


# The main function where we will build the actual app
# SINHLE EDITS
if selected == 'Home':
	st.subheader('Tweets Classifier App')
	image = Image.open('resources/PIC_ART.jpg')
	st.image(image, caption='TEAM JS6', use_column_width=True)
	# st.image(image name, width = None)
	st.write("This is a small nyana paragraph")


if selected == 'Data Analysis':
	st.info("##### For this section we will explore the distribution of the data using different visualisation plots")
	st.write("---")
	bar_graph = Image.open('images/bar_graph.png')
	st.write("##### Bar graph showing tweets per sentiment")
	st.image(bar_graph, width = None)


if selected == 'Models':

	tweet_text = st.text_area('Enter Text (max. 140 characters):') ##user entering a single text to classify and predict
	all_ml_models = ["Logistic Regression", "Random Forest Classifier", "MultinomialNB", "KNeighbors Classifier", "Linear SVC"]
	mode_choice = st.selectbox("Choose Machine Learning Model",all_ml_models)
	st.info("Make a classification using our {} model".format(model_choice))	



if selected == 'Contact Us':
		st.info('### App Developers')
		st.info('Contact details in case you have any query : 🧑🏻‍🤝‍🧑🏽')
		st.write('Mbali')
		st.write('Velly')
		st.write('Andy')
		st.write('Amantle')
		st.write('Mengezi')











		

	


