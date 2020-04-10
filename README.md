# Alitheia 
## (In Greek - Truth)
### Objective: Train Machine Learning models using data from FakeNewsCorpus to generate credibility score for text articles, and to obtain relevant tweets from credible sources. 

### Dataset preparation:
FakeNewsCorpus has 9,408,908 text articles tagged as fake, credible, satire, bias, hate, conspiracy etc in the public version. As data was divided into multiple CSV files, to filter and explore the entire data I created a SQL database using all the csv files. For this project, I filtered 100000 rows tagged as fake and 100000 rows tagged as credible from the database and stored it in csv files.

### Preprocessing:


Data: FakeNewsCorpus (https://github.com/several27/FakeNewsCorpus)

 Required libraries:
 - NumPy
 - Pandas
 - NLTK
 - Scikit-learn
 - Keras
 - Pickle
 - GetOldTweets3 (https://github.com/Mottl/GetOldTweets3)
 
