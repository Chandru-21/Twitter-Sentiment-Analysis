# -*- coding: utf-8 -*-
"""
Created on Wed May 27 16:28:39 2020

@author: Chandramouli
"""

import pandas as pd
from twitterscraper import query_tweets
import datetime as dt
begin_date=dt.date(2020,5,26)
end_date=dt.date(2020,5,27)
limit=1000
lang='english'
#user-chandru21#ts.query.query_tweets_from_user
tweets=query_tweets("Covid19",begindate=begin_date,enddate=end_date,limit=limit,lang=lang)#parsing tweets about covid
df=pd.DataFrame(t.__dict__ for t in tweets)
x=df['text']
x=pd.DataFrame(x)

from langdetect import detect 
def detector(x):
    try:
       return detect(x)
    except:
        None 
x['lang']=x['text'].apply(detector)
#or
x['lang']=x['text'].apply(lambda x:detector(x))

x=x[x['lang']=='en']#taking only english language tweets
x=x.drop(['lang'],axis=1)
x=x.reset_index(drop=True)#resetting index
x.drop_duplicates(subset='text',inplace=True)
x=x.reset_index(drop=True)#resetting index

####
import nltk
nltk.download('punkt')
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')
import numpy as np
import re
import tqdm
import unicodedata

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text
import re
contractions_dict = {
    'didn\'t': 'did not',
    'don\'t': 'do not',
    "aren't": "are not",
    "can't": "cannot",
    "cant": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "didnt": "did not",
    "doesn't": "does not",
    "doesnt": "does not",
    "don't": "do not",
    "dont" : "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i had",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'm": "i am",
    "im": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had",
    "she'd've": "she would have",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they had",
    "they'd've": "they would have",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who's": "who is",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have"
    }

contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))


import tqdm

def expand_contractions(s, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, s)

def remove_stopwords(text, is_lower_case=False, stopwords=None):
    if not stopwords:
        stopwords = nltk.corpus.stopwords.words('english')
    tokens = nltk.word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopwords]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
    
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text


def pre_process_corpus(docs):
    norm_docs = []
    for doc in tqdm.tqdm(docs):
        doc = doc.lower()
        doc = remove_accented_chars(doc)
        doc = expand_contractions(doc)
        doc=remove_stopwords(doc)
        # lower case and remove special characters\whitespaces
        doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
        doc = re.sub(r'www\s+', '', doc)
        doc = re.sub(r'https\s+', '', doc)
        doc = doc.strip()  
        norm_docs.append(doc)
  
    return norm_docs



x1=pre_process_corpus(x['text'])
x1=pd.DataFrame(x1)
x1.columns=['text']

#SENTIMENT PREDICTION
from nltk.corpus import opinion_lexicon
pos_list=set(opinion_lexicon.positive())
neg_list=set(opinion_lexicon.negative())
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import sentiwordnet as swn
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from afinn import Afinn

#USING TEXTBLOB
import textblob

def score(text):
    from textblob import TextBlob
    return TextBlob(text).sentiment.polarity
def predict(text):
    x1['score']=x1['text'].apply(score)
    return(x1)
    
x2=predict(x1)    
x2['Sentiment']=['positive' if score >0 else 'negative' for score in x2['score']]
x2=x2.drop(['score'],axis=1)

x1=x1.drop(['score','Sentiment'],axis=1)
#USING AFINN
afn = Afinn(emoticons = True)
afn.score("I love it")
x_afinn=pd.DataFrame(x1)
x_afinn.columns=['text']
def score(text):
    from afinn import Afinn
    return afn.score(text)
def predict(text):
    x_afinn['score']=x_afinn['text'].apply(score)
    return(x_afinn)
x1_afinn=predict(x_afinn)
x1_afinn['Sentiment']=['positive' if score >0 else 'negative' for score in x1_afinn['score']]
x1_afinn=x1_afinn.drop(['score'],axis=1)

x1=x1.drop(['score','Sentiment'],axis=1)

#sentiment analyzing using vader model
nltk.download('vader_lexicon')
x_vader=pd.DataFrame(x1)
x_vader.columns=['text']
def score(text):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    vader=SentimentIntensityAnalyzer()
    return vader.polarity_scores(text)['compound']
def predict(text):
    x_vader['score']=x_vader['text'].apply(score)
    return(x_vader)
x1_vader=predict(x_vader)
x1_vader['Sentiment']=['positive' if scores>0 else 'negative' for scores in x1_vader['score']]
x1_vader=x1_vader.drop(['score'],axis=1)
x1=x1.drop(['score','Sentiment'],axis=1)



