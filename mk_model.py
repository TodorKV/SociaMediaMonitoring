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
stopwordlist = ['а', 'за', 'горе', 'после', 'повторно', 'не', 
    'сите', 'сум', 'на', 'и', 'било кој', 'се', 'како', 'во', 
    'биде', 'бидејќи', 'било', 'порано', 'да се биде', 'подолу', 'помеѓу', 
    'двајцата', 'од', 'може', 'г', 'направи', 'прави', 'долу', 'за време', 
    'секоја', 'неколку', 'понатаму', 'имаше', 'има', 'тој', 'неа', 'тука', 
    'нејзино', 'сама', 'самиот', 'негов', 'јас', 'ако', 'е', 'тоа', 'него', 
    'само', 'л', 'м', 'ма', 'повеќе', 'повеќето', 'моето', 'сега', 'о', 'еднаш', 
    'или', 'други', 'наши', 'надвор', 'сопствени', 'с', 'исто', 'таа', 'треба', 
    'така', 'некои', 't', 'нивните', 'тогаш', 'таму', 'овие', 'тие', 'ова', 'преку', 
    'до', 'премногу', 'под', 'ве', 'многу', 'беше', 'ние', 'бевме', 'што', 'кога', 
    'каде', 'кое', 'додека', 'кој', 'кого', 'зошто', 'ќе', 'со', 'победи', 'ти', 
    'вие', 'ваши', 'себе']

def preprocess(textdata):
    processedText = []
    
    # Create Lemmatizer and Stemmer.
    wordLemm = WordNetLemmatizer()
    
    # Defining regex patterns.
    urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern       = '@[^\s]+'
    # alhpabet pattern for macedonian alhpabet.
    alphaPattern      = "/^[абвгдѓежзѕијклљмнњопрстќуфхцчџш]+$/"
    sequencePattern   = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"
    
    for tweet in textdata:
        tweet = tweet.lower()
        
        # Replace all URls with 'URL'
        tweet = re.sub(urlPattern,' URL',tweet)
        # Replace all emojis.
        for emoji in emojis.keys():
            tweet = tweet.replace(emoji, "EMOJI" + emojis[emoji])        
        # Replace @USERNAME to 'USER'.
        tweet = re.sub(userPattern,' USER', tweet)        
        # Replace all non alphabets.
        tweet = re.sub(alphaPattern, " ", tweet)
        # Replace 3 or more consecutive letters by 2 letter.
        tweet = re.sub(sequencePattern, seqReplacePattern, tweet)

        tweetwords = ''
        for word in tweet.split():
            # Checking if the word is a stopword.
            #if word not in stopwordlist:
            if len(word)>1:
                # Lemmatizing the word.
                word = wordLemm.lemmatize(word)
                tweetwords += (word+' ')
            
        processedText.append(tweetwords)
        
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


