#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 11:33:25 2020

@author: keerthiraj
"""
#%%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
#%%

# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

#%%

# Scikit-learn
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn import preprocessing

#%%

# NLP 
from nltk.corpus import stopwords
import re

#%%

# =============================================================================
# text cleaner
# =============================================================================

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

backn_1 = re.compile('-\n-')
backn = re.compile('\n')

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
#     text = BeautifulSoup(text, "lxml").text # HTML decoding
    
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

fake = pd.read_csv('all_fake.csv', nrows = 100000)
reliable = pd.read_csv('all_reliable.csv', nrows = 100000)

fr = [fake, reliable]
fakedata = pd.concat(fr)

fakedata['content'] = fakedata['content'].apply(clean_text)

#%%

# =============================================================================
# Machine learning with TF-IDF
# =============================================================================

print('Printing Machine Learning results..... ')

tf = TfidfVectorizer()
x_tf = tf.fit_transform(fakedata['content'])

#%%

lb = preprocessing.LabelBinarizer()
y = lb.fit_transform(fakedata['type'])

#%%

y = y.reshape(len(y),)

x_train, x_test, y_train, y_test = train_test_split(x_tf, y, test_size=0.2, random_state=42)

#%%

# Multinomial Naives Bayes

clf1 = MultinomialNB().fit(x_train, y_train)
predicted= clf1.predict(x_test)

print("MultinomialNB Accuracy:",accuracy_score(y_test, predicted))
print('Classification report is: ', classification_report(y_test, predicted))
print('Confusion matrix is: ', confusion_matrix(y_test, predicted))
print('F1-score is: ', f1_score(y_test, predicted))

#%%

# SGD

clf2 = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None).fit(x_train,y_train)
predicted = clf2.predict(x_test)
print("Linear SVM Accuracy:",accuracy_score(y_test, predicted))
print('Classification report is: ', classification_report(y_test, predicted))
print('Confusion matrix is: ', confusion_matrix(y_test, predicted))
print('F1-score is: ', f1_score(y_test, predicted))

#%%

# =============================================================================
# Deep Learning with Bag of Words
# =============================================================================

print('Printing Deep Learning results..... ')

overall_text = np.array(fakedata['content'].values)
overall_labels = y

#%%

total_word_count = 100000
seq_length = 50 #Number of items in each sequence

#%%

tokenizer = Tokenizer(num_words=total_word_count)
tokenizer.fit_on_texts(overall_text)

sequences = tokenizer.texts_to_sequences(overall_text)
sequences = pad_sequences(sequences, maxlen=seq_length)


#%%
x1_train, x1_test, y1_train, y1_test = train_test_split(sequences, overall_labels, test_size=0.2, random_state=42)

#%%

model = Sequential()
model.add(Embedding(total_word_count, seq_length, input_length=seq_length))
model.add(LSTM(seq_length, dropout=0.3, recurrent_dropout=0.3))
model.add(Dense(20, activation='sigmoid'))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#%%
callbacks = [EarlyStopping(monitor='val_loss', patience=3),
             ModelCheckpoint(filepath='alithea_model.h5', monitor='val_loss', save_best_only=True)]

history = model.fit(x1_train, y1_train, validation_split=0.1, epochs=10, callbacks = callbacks, verbose = 2)

#%%
best_model = load_model('alithea_model.h5')
test_loss, test_accuracy = best_model.evaluate(x1_test, y1_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(test_accuracy))

#%%

model.save("alithea_model.h5")

#%%
bmodel = load_model('alithea_model.h5')
y_pred_lstm = bmodel.predict(x1_test)

#%%

y_pred_labels = []

for pred in y_pred_lstm:
    if pred > 0.5:
        y_pred_labels.append(1)
    else:
        y_pred_labels.append(0)

y_pred_labels = np.array(y_pred_labels, dtype= int)


#%%
print(classification_report(y1_test, y_pred_labels))
print('Confusion matrix is: ', confusion_matrix(y1_test, y_pred_labels))
print('F1-score is: ', f1_score(y1_test, y_pred_labels))

#%%

fpr, tpr, thresholds = roc_curve(y1_test, y_pred_lstm)

plt.plot(fpr, tpr)

#%%

modelname = 'alithea_MNB.sav'
pickle.dump(clf1, open(modelname, 'wb'))


modelname = 'alithea_SGD.sav'
pickle.dump(clf2, open(modelname, 'wb'))

#%%