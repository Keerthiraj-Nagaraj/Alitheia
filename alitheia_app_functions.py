# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 13:39:42 2020

@author: keerthiraj
"""

# Functions for alithea app
import warnings
warnings.filterwarnings("ignore")

# nlp.py
import numpy as np
import pandas as pd
import time
import pickle

import re
from nltk.corpus import stopwords
from nltk.tag import pos_tag

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
    
def username_query_tweets(username, text_query, count):
    # Creation of query object
    tweetCriteria = got.manager.TweetCriteria().setUsername(username)\
                                            .setQuerySearch(text_query)\
                                            .setMaxTweets(count)
    # Creation of list that contains all tweets
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)

    # Creating list of chosen tweet data
    user_tweets = [[tweet.id, tweet.date, tweet.text, tweet.retweets, username] for tweet in tweets]

    # Creation of dataframe from tweets list
    tweets_df = pd.DataFrame(user_tweets, columns = ['Id', 'Datetime', 'Text', 'Retweets', 'Source'])

    # Converting dataframe to CSV
    #tweets_df.to_csv('{}-{}-{}k-tweets.csv'.format(username, text_query, int(count/1000)), sep=',')
    
    return tweets_df

#%%

def credibility_score(sample_text):
    
    # LSTM results
    with open('alitheia_tokenizer.pickle', 'rb') as handle:
        al_tokenizer = pickle.load(handle)
    
    # total_word_count = 100000
    seq_length = 50 #Number of items in each sequence
    
    sample_text = clean_text(sample_text)
    
    text_seq = al_tokenizer.texts_to_sequences(sample_text)
    text_seq_list = [item for sublist in text_seq for item in sublist]
    
    if len(text_seq_list) > 50:

        text_seq_padded = text_seq_list[len(text_seq_list)- seq_length:]
        text_seq_padded_array = np.array(text_seq_padded).reshape(1,seq_length)
        
        bmodel = load_model('alithea_lstm.h5')
        y_pred_lstm = bmodel.predict(text_seq_padded_array)[0][0]
    else:
        y_pred_lstm = 0.0
    
    
    # ML model results - MNB and SGD
    sample_text = [sample_text]
    
    # loading
    with open('alitheia_tfidf_vectorizer.pickle', 'rb') as handle:
        tf_loaded = pickle.load(handle)
    
    text_tfidf = tf_loaded.transform(sample_text)
    
    mnb = pickle.load(open('alithea_MNB.sav', 'rb'))
    y_pred_mnb = mnb.predict_proba(text_tfidf)[0][1]
    
    # sgd = pickle.load(open('alithea_SGD.sav', 'rb'))
    # y_pred_sgd = float(sgd.decision_function(text_tfidf)[0])
    
    y_pred_ml = np.mean([y_pred_mnb, y_pred_lstm])
    cred_score = 100 * y_pred_ml    
    
    return cred_score

#%%
     

def relevant_tweets(sample_text, username = ['nytimes']):
    
    username = ['nytimes', 'washingtonpost', 'politico', 'FinancialTimes', 'NPR']
    count = 5
    
    keytext = text2keywords(sample_text)
    keywords = keytext.split()
    
    overall_tweet_pd = pd.DataFrame(columns = ['Id', 'Datetime', 'Text', 'Retweets', 'Source'])
    
    for i in range(min(len(keywords)-2, 20)):
        
        query = keywords[i] + ' ' +  keywords[i+1] + ' ' +  keywords[i+2]
        
        for user in username:
            usern_df = username_query_tweets(user, query, count)
            overall_tweet_pd = overall_tweet_pd.append(usern_df)
            
    sorted_overall_tweets = overall_tweet_pd.sort_values(by='Retweets', ascending=False)
    
    return_tweets = []
    
    try:
        for i in range(5):
            return_tweets.append( 'Source: ' + sorted_overall_tweets['Source'].values[i] + '\n - Tweet: ' + sorted_overall_tweets['Text'].values[i] + '\n - Retweets: ' + str(sorted_overall_tweets['Retweets'].values[i]) + '\n - Date and time: ' + str(sorted_overall_tweets['Datetime'].values[i]) ) 
    except:
       print('No credible tweets found - try other articles')
    finally:
        for i in range(5):
            return_tweets.append('No credible tweets found - try other articles')

    return return_tweets[:5]


    
#%%    


example_text = """ Pound hits fresh highs against both the Euro and the US Dollar (Tom Holian)

Headline: Bitcoin & Blockchain Searches Exceed Trump! Blockchain Stocks Are Next!

Sterling is now trading at a 9 month high to buy Euros and an 18 month high to buy US Dollars as Sterling appears to be the currency of choice at the moment.

Since the start of the year the Pound has increased by as much as 2.5% against the Euro which is the difference of Â£4,300 on a currency transfer of â‚¬200,000.

Against the US Dollar the Pound has moved by as much as 7% which is the difference of over Â£10,000 on a transfer of USD$200,000 highlighting the importance of keeping up to date with exchange rates on a regular basis particular during this volatile period.

The Dollar has weakened owing to a number of factors and earlier this week US Treasury Secretary Steven Mnuchin has spoken about a â€˜weaker Dollarâ€™.

Indeed, it appears as though the US is not overly concerned with the Dollarâ€™s weakness and this has led to these fresh highs for the Pound vs the Greenback. Great news if youâ€™re considering buying US Dollars at the moment.

Turning the focus towards GBPEUR exchange rates the Pound has made some healthy gains vs the single currency following the release of record low levels of UK unemployment yesterday morning.

As well as this the tone surrounding the topic of Brexit appears to be much more positive at the moment and with phase 2 of the Brexit talks due to start in March could the talks bring us closer to a resolution?

Tomorrow morning brings with the first estimate for UK GDP figures for the final quarter of last year. This could cause a lot of volatility on rates so make sure you keep a close eye out on what happens to Sterling following the release tomorrow morning at 930am. For a free quote call me directly and ask for Tom Holian on 01494787478.

If you have a need to make a currency transfer in the near future then feel free to speak with me directly as I will be more than happy to help you both with trying to time a transaction and getting you the top market rate when you do come to buy your currency compared to your bank or another currency broker.

Even a small improvement in the exchange rates can make a big difference so feel free to to email me and you may find you could save yourself hundreds if not thousands of Pounds. You can email me (Tom Holian) on teh@currencies.co.uk and I will respond to you as soon as I can.

Source: http://www.poundsterlingforecast.com/2018/01/25/pound-hits-fresh-highs-against-both-the-euro-and-the-us-dollar-tom-holian/ """

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

example_text = """Still refusing to face why Donald Trump and the Republicans won in 2016, the national Democratic Party rebuffs proposals from progressives to make the party more democratic and less corporate-dominated, writes Norman Solomon.

By Norman Solomon

With the Democratic Partyâ€™s â€œUnity Reform Commissionâ€ now history, major political forces are entering a new stage of contention over the future of the party. Seven months after the commissionâ€™s first meeting â€” and nine months after Hillary Clinton backer Tom Perez won a close election over Bernie Sanders supporter Keith Ellison to become chair of the Democratic National Committee â€” the battle lines are coming into focus for next year.

The commissionâ€™s final meeting adjourned on Saturday after a few steps toward democratizing the party had won approval â€” due to the grassroots strength of progressives. But the recommendations from the commission will go to the Rules and Bylaws Committee, which was one of the DNC decision-making bodies that Perez subjected to a purge two months ago. Now, in the words of Jim Zogby (who was removed from the Executive Committee by Perez), â€œThere are virtually no Bernie supporters on the Rules and Bylaws Committee.â€

When the latest Unity Reform Commission meeting got underway, Perez talked a lot about unity. But kicking Sanders supporters off of key DNC committees is the ugly underside of an ongoing dual discourse. (Are we supposed to believe Perezâ€™s soothing words or our own eyes?) And party unity behind a failed approach â€” internally undemocratic and politically hitched to corporate wagons â€” would hardly be auspicious.

â€œEmerging sectors of the electorate are compelling the Democratic Party to come to terms with adamant grassroots rejection of economic injustice, institutionalized racism, gender inequality, environmental destruction and corporate domination,â€ says the recent report â€œAutopsy: The Democratic Party in Crisisâ€ (which I co-authored). The report adds: â€œSiding with the people who constitute the base isnâ€™t truly possible when party leaders seem to be afraid of them.â€

DNC Chairman Perez and allied power brokers keep showing that theyâ€™re afraid of the partyâ€™s progressive base. No amount of appealing rhetoric changes that reality.

â€œWe pride ourselves on being inclusive and welcoming to all,â€ the Democratic National Committee proclaimed anew at the start of this month, touting the commission meeting as â€œopen to the public.â€ Yet the DNC delayed and obscured information about the meeting, never replying to those who filled out an online RSVP form â€” thus leaving them in the dark about the times of the meeting. In short, the DNC went out of its way to suppress public turnout rather than facilitate it.

Rebuking the DNC

One member of the task force that wrote the Autopsy, Karen Bernal, is the chair of the Progressive Caucus of the California Democratic Party. After traveling across the country and sitting in the sparse audience during the first day of the Unity Reform Commission meeting, she took the liberty of speaking up as the second day got underway. Bernal provided a firm rebuke of the DNCâ€™s efforts to suppress public attendance.

â€œFor all of the talk about wanting to improve and reform and make this party more transparent, the exact opposite has happened,â€ Bernal told the commission. (Her intervention, which lasted a little more than two minutes, aired in full on C-SPAN.)

On Sunday, a mass email from Zogby via Our Revolution summed up: â€œWe are fighting for racial, social, economic, and environmental justice. The Democratic Party needs everyone, regardless of their race, religion, gender, sexual orientation, age, ability, country of origin, language, or socioeconomic status, to be deeply involved in order to change the course of this country.â€

For those reasons, he added, â€œwe are calling for an end to superdelegates, [for] open primaries and caucuses, [for] same-day registration, and [for] more transparent, fair, and accountable leadership at the helm of the DNC.â€

Overall, the commission approved some recommendations that were partial victories for progressives. Among the most notable: It called for reducing the number of notoriously undemocratic superdelegates to the national convention from 712 to about 300, while the only democratic number would be zero. [Superdelegates are party insiders who are not chosen through a primary or caucus but nevertheless get to vote for the partyâ€™s nominees. In 2016, they broke overwhelmingly for Hillary Clinton.]

The commission somewhat improved transparency for often-dubious DNC contracts with high-paid consultants and vendors, while defeating sensible amendments by commission member Nomiki Konst â€” who spoke with notable clarity about the need to clamp down on financial conflicts of interest among DNC decision-makers.

The eight Sanders appointees â€” Konst, Zogby, Larry Cohen, Lucy Flores, Jane Kleeb, Gus Newport, Nina Turner and Jeff Weaver â€” put up a good fight as members of the Unity Reform Commission. They were outnumbered, and on key issues were often outvoted, by the 13 whoâ€™d been selected by Clinton or Perez. Next year, the odds to overcome will be much worse.

With the purged Rules and Bylaws Committee now overwhelmingly stacked against progressives, only massive pressure from the grassroots will be able to sustain momentum toward a democratic Democratic Party. Meanwhile, corporate forces will do all they can to prevent the Democratic Party from living up to its first name.

Norman Solomon, the national coordinator of the online activist group RootsAction.org, is a member of the task force that wrote â€œAutopsy: The Democratic Party in Crisis.â€ His books include War Made Easy: How Presidents and Pundits Keep Spinning Us to Death.
"""

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