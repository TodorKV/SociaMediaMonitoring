# utilities
import re
import pickle
import numpy as np
import pandas as pd
import nltk

# nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')

# sklearn
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report

# Defining dictionary containing all emojis with their meanings.
emojis = {':)': 'насмевка', ':-)': 'насмевка', ';d': 'намигнување', ':-E': 'вампир', ':(': 'тажно',
          ':-(': 'тажно', ':-<': 'тажно', ':P': 'малина', ':O': 'изненадена',
          ':-@': 'шокиран', ':@': 'шокиран',':-$': 'збунет', ':\\': 'изнервиран',
          ':#': 'неми', ':X': 'неми', ':^)': 'насмевка', ':-&': 'збунет', '$_$': 'лаком',
          '@@': 'превртување на очите', ':-!': 'збунето', ':-D': 'насмевка', ':-0': 'викна', 'O.o': 'збунето',
          '<(-_-)>': 'робот', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'намигнување',
          ';-)': 'намигнување', 'О:-)': 'ангел','О*-)': 'ангел','(:-D': 'озборувања', '=^.^=' : 'мачка'}

## Defining set containing all stopwords in english.
stopwordlist = ['и', 'или', 'но', 'а', 'со', 'до', 'за', 'во', 'од', 'на', 'не', 'се', 'ќе', 'ме', 'те', 'го', 'ја', 'ни', 'нека', 'би', 'да', 'си', 'ве', 'им', 'му', 'ѝ', 'им', 'нив', 'многу', 'повеќе', 'најмногу', 'меѓутоа', 'односно', 'па', 'при', 'што', 'така', 'токму', 'туку', 'џабе']

def preprocess(textdata):
    processedText = []
    
    # Create Lemmatizer and Stemmer.
    wordLemm = WordNetLemmatizer()
    
    # Defining regex patterns.
    urlPattern = r'(https?:\/\/(?:www\.|(?!www))[^\s\.]+\.[^\s]{2,}|www\.[^\s]+\.[^\s]{2,})'
    # urlPattern        = r'https?://[A-Za-z0-9./]+'
    userPattern       = r'@\w+'
    # alhpabet pattern for macedonian alhpabet.
    alphaPattern      = r"[^абвгдѓежзѕијклљмнњопрстќуфхцчџш]+"
    sequencePattern   = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"
    
    for tweet in textdata:
        tweet = tweet.lower()
        print(tweet)
        # Replace all URls with 'URL'
        tweet = re.sub(urlPattern,' ЛИНК',tweet)
        print(tweet)

        # Replace all emojis.
        for emoji in emojis.keys():
            tweet = tweet.replace(emoji, "ЕМОТИКОНА" + emojis[emoji])
        print(tweet)     
        # Replace @USERNAME to 'USER'.
        tweet = re.sub(userPattern,' ПОТРЕБИТЕЛ', tweet)
        print(tweet)        
        # Replace all non alphabets.
        tweet = re.sub(alphaPattern, " ", tweet)
        print(tweet)
        # Replace 3 or more consecutive letters by 2 letter.
        tweet = re.sub(sequencePattern, seqReplacePattern, tweet)
        print(tweet)
        print("------------------")
        tweetwords = ''
        for word in tweet.split():
            if word not in stopwordlist:
                if len(word)>1:
                    # Lemmatizing the word.
                    word = wordLemm.lemmatize(word)
                    tweetwords += (word+' ')
            
        processedText.append(tweetwords)
    print("Processed Text: ")
    print(processedText)
    return processedText


def load_models():
    '''
    Replace '..path/' by the path of the saved models.
    '''
    
    # Load the vectoriser.
    file = open('vectoriser-ngram-(1,2).pickle', 'rb')
    vectoriser = pickle.load(file)
    file.close()
    # Load the LR Model.
    file = open('Sentiment-LR.pickle', 'rb')
    LRmodel = pickle.load(file)
    file.close()
    
    return vectoriser, LRmodel

def predict(vectoriser, model, text):
    # Predict the sentiment
    textdata = vectoriser.transform(preprocess(text))
    sentiment = model.predict(textdata)
    
    # Make a list of text with sentiment.
    data = []
    for text, pred in zip(text, sentiment):
        data.append((text,pred))
        
    # Convert the list into a Pandas DataFrame.
    df = pd.DataFrame(data, columns = ['text','sentiment'])
    df = df.replace([2,1], ["Negative","Positive"])
    return df


