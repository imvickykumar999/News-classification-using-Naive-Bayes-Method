
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups

data=fetch_20newsgroups()
categories=data.target_names

train=fetch_20newsgroups(subset='train',categories=categories)
test=fetch_20newsgroups(subset='test',categories=categories)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

model=make_pipeline(TfidfVectorizer(),MultinomialNB())
model.fit(train.data,train.target)
labels=model.predict(test.data)

from sklearn.metrics import confusion_matrix
mat=confusion_matrix(test.target,labels)

def predict_category(s,train=train,model=model):
    pred=model.predict([s])
    print(pred)
    return train.target_names[pred[0]]

predict_category('''
Determined to continue Gaza operation until aim met: Israeli PM
''') 
