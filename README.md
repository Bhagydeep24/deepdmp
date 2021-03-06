# Python Flask App to Predict the Movie Rating based on the user Comment

So this project is a part of my Data mining final project where we were suppose to participate in the Kaggle competition of predicting the 
movie rating using the comments provided.

Link to Kaggle Competition: <!-- Links -->https://www.kaggle.com/jvanelteren/boardgamegeek-reviews

Let us begin with building our model and for that I have used model.py

### 1. Import all the required files.

```
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
```
### 2.Get our data

Set the path of your csv

```
path = '../boardgamegeek-reviews/bgg-13m-reviews.csv';
```

Now one of the biggest problem is the size of the dataset which is above 1GB. Due that reason we are going to use chunks of 1000000 records 
to read and finally join them into the dataframe.

```
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
```

Now you you would have seen .feather file it is the dataset that we read in previous step. We will take the dataframe created 
in the previous step and convert into feather format. We are converting into feather because accessing data using feather format 
is lot easier and we don't need to call our csv file again and again. Once you write the below code for convertion it will automatically
create feather file for us.

```
train_data_frame.to_feather('movie_data_raw.feather')# storing pandas dataframe into movie_data_raw feather format

train_df_n = pd.read_feather('movie_data_raw.feather')# store dataframe from movie_data_raw feather format
```

### 3. Data Cleaning

Whatever data we get is not clean and we need to clean according to our need. Like remove unwanted feathers, remove records with 
null values, convert all text to lower, remove punctuations, numeric values, extra space, html tags.

```
train_df_n = train_df_n.drop(['Unnamed: 0','user'], axis=1) #remove columns 0 and user

train_df_n = train_df_n.drop(['ID','name'], axis=1) #remove columns ID and name

train_df_n.dropna(inplace=True) #remove null records

train_na = train_df_n.to_numpy() #convert dataframe to ndarray

for x in range(len(train_na[:,1])):
  txt=str(train_na[x][1])
  html_tags=re.compile('<.*?>');  
  txt=re.sub(html_tags, '', txt);#remove html tags
  txt=re.sub(r"[^a-zA-Z0-9']+", ' ', txt);#just keep chars and number and remove rest all punctions
  train_na[x][1]=txt
```

### 4. Tokenization, TF-IDF, and Model Generation

We are going to use pipeline and MultinomialNM for generating our model but before that we tokenize and later TF-IDF our traing 
data for the generation of model

```
#Split dataset in training and test in the portion of 80-20 respectively.
X_train, X_test, y_train, y_test = train_test_split(train_na[:,1], train_na[:,0], test_size=0.20)


#Vectorization
vector_form2=CountVectorizer()
Xtrain_tf=vector_form2.fit_transform(X_train)

vector_form3=TfidfTransformer()
Xtrain_tfid=vector_form3.fit_transform(Xtrain_tf)

y_train=y_train.astype(int)
y_test=y_test.astype(int)

m1 = MultinomialNB().fit(Xtrain_tfid, y_train)

#though we have generated token, TF-IDF and created model m1 but we are going to use pipleline for prediction

pipeline = Pipeline([
    ('count_vectorizer', CountVectorizer(lowercase = True, stop_words = stopwords.words('english'))), 
    ('tfidf_transformer',  TfidfTransformer()), #weighs terms by importance to help with feature selection
    ('classifier', MultinomialNB()) ])

```

Once our model is created we save it in pickle format it is just like feather. And later model.pkl will be used to predict 
our rating. We won't need to read our dataset and generate model once our model.pkl file is generated. Just like feather we don't
need to do anything to create model.pkl externally. We just need to run the below code and model.pkl will be automatically genearated.

```
# Saving model to disk
pickle.dump(m1, open('model.pkl','wb'))
```

Now our Model is generated. Let us move to front page, it is available in templates/index.html. We don't need to do much in it. Just a form with text box and submit button on which if clicked it will return the rating based on the text written in the text box.

![](app_pic.png)
</br>

To run the app go to your terminal and make sure your current directory is your project directory and then type python app.py

It will start your app on http://127.0.0.1:5000/ 

### Challenge and how to upload on live server.

The challenging part was to not build the app but to upload it on the server. There are many solution availbale online but none of them have model which which above 100mb. One of the solution to go live with your your flask project is through heroku and I used the same but it uses GitHub to upload your project. And our model.pkl is about 120 MB and Git don't allow us to upload file above 25 MB. 

So the solution to upload file on GitHub which is above 25 MB is using Git LFS
<!-- Links -->
https://git-lfs.github.com/

Download it on your PC and once you are done move to next step. Luckily I found a life saving video is short time though it was in Thai but I just followed the instruction displayed on screen. Here is the link : <!-- Links -->https://www.youtube.com/watch?v=84nRMsH2L7o&list=LLwUe5wS-oG6otJTEDk2OU0g&index=2&t=307s

I had little issue in this as well while pushing the pkl file but I tried to push it forcefully with code "git push origin master --force" the only to note here it is a bad habit to do so it will remove all your existing file.

```
(base) Deeps-MacBook-Air2:DMProject deeppatel$ git init
(base) Deeps-MacBook-Air2:DMProject deeppatel$ git remote add origin <your git repository link>
(base) Deeps-MacBook-Air2:DMProject deeppatel$ git lfs track "*.pkl"
Tracking "*.pkl"
(base) Deeps-MacBook-Air2:DMProject deeppatel$ git add model.pkl
(base) Deeps-MacBook-Air2:gitfolder deeppatel$ git commit -m "commit one"
[master (root-commit) 2dc7a2d] commit one
 1 file changed, 3 insertions(+)
 create mode 100644 model.pkl
(base) Deeps-MacBook-Air2:gitfolder deeppatel$ git push origin master --force
Uploading LFS objects: 100% (1/1), 125 MB | 0 B/s, done.   
```
### Upload on Heroku

Here is the link I followed while uploading: </br>
Link 1: <!-- Link -->https://www.youtube.com/watch?v=bjsJOl8gz5k&list=PLZoTAELRMXVOAvUbePX1lTdxQR8EY35Z1</br>
And as we used lfs we need to add build pack to our heroku app and generate token</br>
Link 2: <!-- Link -->https://elements.heroku.com/buildpacks/raxod502/heroku-buildpack-git-lfs</br>
Link 3: <!-- Link -->https://towardsdatascience.com/everything-from-your-deep-learning-model-to-a-web-app-279cd733f3d4


