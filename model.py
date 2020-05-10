# Importing the libraries
import pandas as pd 
import dask.dataframe as dd
import os
from tqdm import tqdm
import pickle
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')

TRAIN_PATH = '../boardgamegeek-reviews/bgg-13m-reviews.csv';


# Assume we only know that the csv file is somehow large, but not the exact size
# we want to know the exact number of rows

# Method 1, using file.readlines. Takes about 20 seconds.
# with open(TRAIN_PATH) as file:
#     n_rows = len(file.readlines())

# print (f'Exact number of rows: {n_rows}')

# traintypes = {'Unnamed: 0': 'int64',
#               'user': 'str', 
#               'rating': 'float32',
#               'comment': 'str',
#               'ID': 'int64',
#               'name': 'str'}

# cols = list(traintypes.keys())

# chunksize = 1_000_000

# df_list = [] # list to hold the batch dataframe

# for df_chunk in tqdm(pd.read_csv(TRAIN_PATH, usecols=cols, dtype=traintypes, chunksize=chunksize)):
#   df_list.append(df_chunk)
  
# train_df = pd.concat(df_list)

# del df_list

# train_df.to_feather('movie_data_raw.feather')

#train_df_new = pd.read_feather('movie_data_raw.feather')

# train_df_new = train_df_new.drop(['Unnamed: 0','user','ID','name'], axis=1)

# train_df_new.dropna(inplace=True)

#X_train1, X_test1, y_train1, y_test1 = train_test_split(train_df_new.comment, train_df_new.rating, test_size=0.20)

# pipeline = Pipeline([
#     ('count_vectorizer', CountVectorizer(lowercase = True, stop_words = stopwords.words('english'))), 
#     ('tfidf_transformer',  TfidfTransformer()), #weighs terms by importance to help with feature selection
#     ('classifier', MultinomialNB()) ])

# pipeline.fit(X_train1, y_train1.astype('int'))

# # Saving model to disk
# pickle.dump(pipeline, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print("The prediction is:")
print(model.predict(["good movie"]))