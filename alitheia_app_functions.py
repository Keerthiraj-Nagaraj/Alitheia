# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 13:39:42 2020

@author: keert
"""

# Functions for alithea app
import warnings
warnings.filterwarnings("ignore")

# nlp.py
import numpy as np
import pandas as pd
import time

import re
from nltk.corpus import stopwords
from nltk.tag import pos_tag

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

import GetOldTweets3 as got

# =============================================================================
# Credibility application
# =============================================================================

#%%

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
backn_1 = re.compile('-\n-')
backn = re.compile('\n')

def clean_text(text):
    try:
        text = text.lower() # lowercase text
    except:
        print('sample is float')
    finally:
        text = str(text).lower()
    
    text = backn_1.sub('', text)
    text = backn.sub(' ', text)
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text

    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    
    return text  
    
#%%
    
def text2keywords(text):
    try:
        text = text.lower() # lowercase text
    except:
        print('sample is float')
    finally:
        text = str(text).lower()
    
    text = backn_1.sub('', text)
    text = backn.sub(' ', text)
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = [word for word in text.split() if word not in STOPWORDS] # delete stopwors from text
    
    tagged_words = pos_tag(text)

    nouns = [word for word,pos in tagged_words if pos == 'NN']
    
    uniqueWords = [] 
    for i in nouns:
          if not i in uniqueWords:
              uniqueWords.append(i);
        
    atext = " "
    for n in nouns:
        if len(n) > 3:
            atext += n
            atext += " "    
    
    return atext

#%%
    
def username_query_tweets_to_csv(username, text_query, count):
    # Creation of query object
    tweetCriteria = got.manager.TweetCriteria().setUsername(username)\
                                            .setQuerySearch(text_query)\
                                            .setMaxTweets(count)
    # Creation of list that contains all tweets
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)

    # Creating list of chosen tweet data
    user_tweets = [[tweet.date, tweet.text, tweet.hashtags, tweet.retweets] for tweet in tweets]

    # Creation of dataframe from tweets list
    tweets_df = pd.DataFrame(user_tweets, columns = ['Datetime', 'Text', 'Hashtags', 'Retweets'])

    # Converting dataframe to CSV
    #tweets_df.to_csv('{}-{}-{}k-tweets.csv'.format(username, text_query, int(count/1000)), sep=',')
    
    return tweets_df

#%%

def credibility_score(sample_text):

    fake = pd.read_csv('all_fake.csv', nrows = 1000)
    reliable = pd.read_csv('all_reliable.csv', nrows = 1000)
    
    fr = [fake, reliable]
    fakedata = pd.concat(fr)
    fakedata['content'] = fakedata['content'].apply(clean_text)
    
    total_word_count = 10000
    seq_length = 50 #Number of items in each sequence
    
    overall_text = np.array(fakedata['content'].values)
    tokenizer = Tokenizer(num_words=total_word_count)
    tokenizer.fit_on_texts(overall_text)
    
    sequences = tokenizer.texts_to_sequences(overall_text)
    sequences = pad_sequences(sequences, maxlen=seq_length)
    
    atext = clean_text(sample_text)
    tokenizer = Tokenizer(num_words=total_word_count)
    text_seq = tokenizer.texts_to_sequences(atext)
    text_seq_padded =  pad_sequences(text_seq, maxlen=seq_length)
    
    bmodel = load_model('alithea_lstm.h5')
    
    y_pred_lstm = bmodel.predict(text_seq_padded)
    cred_score = 100 * np.mean(y_pred_lstm)
    
    return cred_score


#%%   

def relevant_tweets(sample_text, source = 'nytimes'):
    
    username = source
    count = 1
    
    all_tweets = []
    
    keytext = text2keywords(example_text)
    
    keywords = keytext.split()
    
    for word in keywords:
        
        query = word
    
        usern_df = username_query_tweets_to_csv(username, query, count)
        curr_tweets = list(usern_df['Text'].values)
        
        all_tweets.extend(curr_tweets)
    
    return all_tweets



#%%

example_text = "In the weeks leading up to the 2016 US presidential election, you’ll likely remember encountering stories about the Pope endorsing Donald Trump or Hillary Clinton selling weapons to ISIS. Fake news stories like these are not new, but in the post-print social media age, the potential for misleading information to go viral is. This topic guide will provide you with tools, strategies, and additional resources to help you cultivate informed skepticism about the information you encounter on the Internet, and shield yourself from the dangers of consuming and sharing dubious or flat out incorrect information."

print(example_text)

#%%

t1 = time.time()
credscore = credibility_score(example_text)
print('Credibility score is ', credscore)
t2 = time.time()
print("Credibility score time in seconds", t2-t1)

#%%

t1 = time.time()
rel_tweets = relevant_tweets(example_text)
print('Relevant tweets are ', rel_tweets)
t2 = time.time()
print("relevant_tweets time in seconds", t2-t1)

#%%


