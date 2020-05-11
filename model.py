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
import re
from nltk.corpus import stopwords
nltk.download('stopwords')
stop = stopwords.words('english')

path = '../boardgamegeek-reviews/bgg-13m-reviews.csv';

#Now let's set chucnk size equal to 1 million
chunk_size = 1_000_000

# Let us set the column name for our dataframe
train_types = {'Unnamed: 0': 'int64',
              'user': 'str', 
              'rating': 'float32',
              'comment': 'str',
              'ID': 'int64',
              'name': 'str'}
columns = list(train_types.keys())

data_list = [] # list to store chunks

for data_chunk in tqdm(pd.read_csv(path, usecols=columns, dtype=train_types, chunksize=chunk_size)):
  data_list.append(data_chunk)

train_data_frame = pd.concat(data_list) #Here we are merging all the data frames in one

del data_list # To avoid unwanted space occupation let us release data_list

train_data_frame.to_feather('movie_data_raw.feather')# storing pandas dataframe into movie_data_raw feather format

train_df_n = pd.read_feather('movie_data_raw.feather')# store dataframe from movie_data_raw feather format
train_df_n.head()

train_df_n = train_df_n.drop(['Unnamed: 0','user'], axis=1)

train_df_n = train_df_n.drop(['ID','name'], axis=1)

train_df_n.dropna(inplace=True)

train_na = train_df_n.to_numpy()

for x in range(len(train_na[:,1])):
  txt=str(train_na[x][1])
  html_tags=re.compile('<.*?>');
  txt=re.sub(html_tags, '', txt);#remove html tags
  txt=re.sub(r"[^a-zA-Z0-9']+", ' ', txt);#just keep chars and number and remove rest all punctions
  train_na[x][1]=txt
  
X_train, X_test, y_train, y_test = train_test_split(train_na[:,1], train_na[:,0], test_size=0.20)


#Vectorization
vector_form2=CountVectorizer()
Xtrain_tf=vector_form2.fit_transform(X_train)

vector_form3=TfidfTransformer()
Xtrain_tfid=vector_form3.fit_transform(Xtrain_tf)

y_train=y_train.astype(int)
y_test=y_test.astype(int)

m1 = MultinomialNB().fit(Xtrain_tfid, y_train)

pipeline = Pipeline([
    ('count_vectorizer', CountVectorizer(lowercase = True, stop_words = stopwords.words('english'))), 
    ('tfidf_transformer',  TfidfTransformer()), #weighs terms by importance to help with feature selection
    ('classifier', MultinomialNB()) ])

# Saving model to disk
pickle.dump(m1, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print("The prediction is:")
print(model.predict(["good movie"]))