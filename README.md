# Twitter-Sentiment-Analysis-NLP-twitterscraper-

OBJECTIVE:
Predicting the sentiment of various tweets in an hashtag using text analytics techniques

DATA:
Getting data without using API -- TwitterScrapper Library
Then query tweets from an Hashtag eg,

from twitterscraper import query_tweets
lang='english'
tweets=query_tweets("Covid19",begindate=begin_date,enddate=end_date,limit=limit,lang=lang)

PREPROCESSING:

Using NLTK library I did the following things,
Removing Stopwords-Stopwords are the words in any language which does not add much meaning to a sentence. They can safely be ignored without sacrificing the meaning of the sentence eg-“the”, “a”, “an”, “in"

Stemming,Lemmatization-Reduce a word to its root or base unit eg:eating, eats, eaten root verb is eat.

Removing accented characters-eg-Café,Naïve

Expanding contractions-eg-Could've to could have

Removing unwanted characters using RE -eg removing emoticons,url,html tags etc..

UNSUPERVISED LEARNING:

Using affin,vader,textblob predicting the sentiment of the tweets ,whether they are of Postive Sentiment or Negative Sentiment.


