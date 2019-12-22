from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import pickle

data = pd.read_csv(r"E:\Datasets\DataSet\spam-text-message-classification\SPAM.csv")
#Data cleaning and preprocessing
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

corpus = []
for i in range(0, len(data)):
    mail = re.sub('[^a-zA-Z]', ' ', data['Message'][i])
    mail = mail.lower()
    mail = mail.split()   
    mail = [ps.stem(word) for word in mail if not word in stopwords.words('english')]
    mail = ' '.join(mail)
    corpus.append(mail)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

#splitting the data
y=pd.get_dummies(data['Category'])
y=y.iloc[:,1].values

# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training model using Naive bayes classifier
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)
y_pred=spam_detect_model.predict(X_test)

spam_detect_model.score(X_train,y_train)

#pickle file
pickle.dump(spam_detect_model, open('nlp_model.pkl', 'wb'))
pickle.dump(cv, open('transform.pkl', 'wb'))
