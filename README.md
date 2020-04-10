# Alitheia 

### Objective: Train Machine Learning models using data from FakeNewsCorpus to generate credibility score for text articles, and to obtain relevant tweets from credible sources. 

### Dataset preparation:

Data: FakeNewsCorpus (https://github.com/several27/FakeNewsCorpus)

FakeNewsCorpus has 9,408,908 text articles tagged as fake, credible, satire, bias, hate, conspiracy etc in the public version. As data was divided into multiple CSV files, to filter and explore the entire data I created a SQL database using all the csv files. For this project, I randomly selected 100000 rows tagged as fake and 100000 rows tagged as credible from the database and stored it in csv files.

## alithea_model_training.py:

### Preprocessing:

- For each text article, I perform text cleaning by dealing with special characters, numbers, english stopwords, line spaces, capital letters. 
- Cleaned text is converted into numerical vectors using TF-IDF vectorizer from scikit-learn for machine learning model training. 
- Cleaned text is converted into numerical vector by using tokenizer and sequence padding from Keras library for deep learning model training.
- Data is split into training and testing sets.
- Key words (Noun phrases) are extracted for tweet extraction.

### Models

- Machine learning - Multinomial Naives Bayes
- Deep learning - Long Short Term Memory Neural Networks

Trained models and preprocessing models (tfidf vectorizer and tokenizer) are saved in pickle format.

## alithea_app_functions.py

Credibility score - credibility_score(sample_text)
- Takes text (string format) as input and outputs credibility score (0-100) based on the prediction probabiliy values from saved machine learning and deep learning models

Relevant tweets extracter - relevant_tweets(sample_text, username = ['nytimes'])
- Takes text (string format) as input and outputs 5 most popular relevant tweets from credible sources such as The New York times, The Washington Post, National Public Radio, Poilitico etc 

### Required libraries:
 - NumPy
 - Pandas
 - NLTK
 - Scikit-learn
 - Keras
 - Pickle
 - GetOldTweets3 (https://github.com/Mottl/GetOldTweets3)
 
