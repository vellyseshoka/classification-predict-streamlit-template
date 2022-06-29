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





df = pd.read_csv('resources/train.csv')


def getAnalysis(sentiment):
    if sentiment == 0:
        return 'Neutral'
    elif sentiment == 1:
        return 'Pro'
    elif sentiment == -1:
        return 'Anti'
    else: 
        return 'News'
df['analysis'] = df['sentiment'].apply(getAnalysis)





def frequency(tweet):
    
    """
    This function determines the frequency of each word in a collection of tweets 
    and stores the 25 most frequent words in a dataframe, 
    sorted from most to least frequent     
    """
    cv = CountVectorizer(stop_words='english')                             # Count vectorizer excluding english stopwords
    words = cv.fit_transform(tweet)
    
    # Count the words in the tweets and determine the frequency of each word
    sum_words = words.sum(axis=0)
    words_freq = [(word, sum_words[0, i]) for word, i in cv.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    
    # Create a dataframe to store the top 25 words and their frequencies
    frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])
    frequency = frequency.head(20)
    
    return frequency

# Extract the top 25 words in each class
pro_frequency = frequency([df['sentiment']==1])
anti_frequency = frequency([df['sentiment']==-1])
news_frequency = frequency([df['sentiment']==2])
neutral_frequency = frequency([df['sentiment']==0])






st.set_option('deprecation.showPyplotGlobalUse', False)

st.sidebar.title('Options')

option = st.sidebar.selectbox('Which Dashboard?', ('Home','Predictions', 'Charts', 'Raw Tweets','Contact Us'))

st.header(option)





if option == 'Home':
    st.subheader('Tweets Classifier App')
    from PIL import Image
    image = Image.open('resources/imgs/GLOBAL-WARMING.jpg')
    st.image(image, caption='Which Tweet are you?', use_column_width=True)





vectorizer = open('resources/tfidfvect.pkl','rb')   ##  will be replaced by the cleaning and preprocessing function
tweet_cv = joblib.load(vectorizer)

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
        all_ml_models = ["lr","nb","rf","knn", "EDSA", "lsvc_op"]
        model_choice = st.selectbox("Choose ML Model",all_ml_models)

        prediction_labels = {'Negative':-1,'Neutral':0,'Positive':1,'News':2}


        if st.button('Classify'):

            st.text("Original test :\n{}".format(tweet_text))

            if model_choice == 'lr':
                prediction = use_model("lr.pkl",tweet_text )
            elif model_choice == 'rf':
                prediction = use_model("rf.pkl",tweet_text )
            elif model_choice == 'nb':
                prediction = use_model("nb.pkl",tweet_text )
            elif model_choice == 'knn':
                prediction = use_model("knn.pkl",tweet_text )
            elif model_choice == 'EDSA':
                prediction = use_model("Logistic_regression.pkl",tweet_text )        
            elif model_choice == 'lsvc_op':
                prediction = use_model("lsvc_op.pkl",tweet_text )

             
            final_result = get_keys(prediction,prediction_labels)
            st.success("Tweets Categorized as: {}".format(final_result))




if option == 'Charts':
    st.subheader('Lets Visualize')
    
    #Show the value counts

    df['analysis'].value_counts()

    #plot and visualize the counts
    plt.title('Sentiment Analysis')
    plt.xlabel('Sentiment')
    plt.ylabel('Counts')
    df['analysis'].value_counts().plot(kind='bar')
    st.pyplot()

    style.use('seaborn-pastel')

    fig, axes = plt.subplots(ncols=2, 
                            nrows=1, 
                            figsize=(20, 10), 
                            dpi=100)

    sns.countplot(df['sentiment'], ax=axes[0])

    labels=['Pro', 'News', 'Neutral', 'Anti'] 

    axes[1].pie(df['sentiment'].value_counts(),
                labels=labels,
                autopct='%1.0f%%',
                shadow=True,
                startangle=90,
                explode = (0.1, 0.1, 0.1, 0.1))

    fig.suptitle('Tweet distribution', fontsize=20)
    st.pyplot()
    


    # Plot the distribution of the length tweets for each class using a box plot
    sns.boxplot(x=df['sentiment'], y=df['length'], data=df, palette=("Blues_d"))
    plt.title('Tweet length for each class')
    st.pyplot()

    # Extract the words in the tweets for the pro and anti climate change classes 
    anti_words = ' '.join([twts for twts in anti_frequency['word']])
    pro_words = ' '.join([twts for twts in pro_frequency['word']])
    news_words = ' '.join([twts for twts in news_frequency['word']])
    neutral_words = ' '.join([twts for twts in neutral_frequency['word']])

    # Create wordcloud for the anti climate change class
    anti_wordcloud = WordCloud(width=800, 
                            height=500, 
                            random_state=110, 
                            max_font_size=110, 
                            background_color='white',
                            colormap="Reds").generate(anti_words)

    # Create wordcolud for the pro climate change class
    pro_wordcloud = WordCloud(width=800, 
                          height=500, 
                          random_state=73, 
                          max_font_size=110, 
                          background_color='white',
                          colormap="Greens").generate(pro_words)

    # Create wordcolud for the news climate change class
    news_wordcloud = WordCloud(width=800, 
                            height=500, 
                            random_state=0, 
                            max_font_size=110, 
                            background_color='white',
                            colormap="Blues").generate(news_words)

    # Create wordcolud for the neutral climate change class
    neutral_wordcloud = WordCloud(width=800, 
                            height=500, 
                            random_state=10, 
                            max_font_size=110, 
                            background_color='white',
                            colormap="Oranges").generate(neutral_words)



    # Plot pro and anti wordclouds next to one another for comparisson
    f, axarr = plt.subplots(2,2, figsize=(35,25))
    axarr[0,0].imshow(pro_wordcloud, interpolation="bilinear")
    axarr[0,1].imshow(anti_wordcloud, interpolation="bilinear")
    axarr[1,0].imshow(neutral_wordcloud, interpolation="bilinear")
    axarr[1,1].imshow(news_wordcloud, interpolation="bilinear")

    # Remove the ticks on the x and y axes
    for ax in f.axes:
        plt.sca(ax)
        plt.axis('off')

    axarr[0,0].set_title('Pro climate change\n', fontsize=35)
    axarr[0,1].set_title('Anti climate change\n', fontsize=35)
    axarr[1,0].set_title('Neutral\n', fontsize=35)
    axarr[1,1].set_title('News\n', fontsize=35)
    #plt.tight_layout()
    st.pyplot()

    st.write(pro_frequency.tail())







if option == 'Raw Tweets':
    st.subheader('Twitter Dashboard')
    #print all postive tweets
    j=1
    sortedDF = df.sort_values(by=['polarity'])
    for i in range(0, sortedDF.shape[0]):
        if(sortedDF['analysis'][i]=='Pro'):
            print(str(j)+') ' +sortedDF['message'][i])
            print()
            j = j+1


if option == 'Contact Us':
    st.subheader('Reach the App Developers here:')
    st.info('Contact details in case you any query or would like to know more of our designs:')
    st.write('Nomvuselelo Simelane: one@gmail.com')
    st.write('Thobekani Masondo: two@gmail.com')
    st.write('Ndamulelelo: three@gmail.com')
    st.write('Sandra Malope: fourl@gmail.com')
    st.write('Namhla:: five2@gmail.com')
    st.write('John Sekgobela: jrsmsekgobela@gmail.com')

    # Footer 
    image = Image.open('resources/imgs/EDSA_logo.png')

    st.image(image, caption='Team ZM3', use_column_width=True)

