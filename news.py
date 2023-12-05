
from sklearn.datasets import fetch_20newsgroups
import pickle

data=fetch_20newsgroups()
categories=data.target_names

train=fetch_20newsgroups(subset='train',categories=categories)
# test=fetch_20newsgroups(subset='test',categories=categories)

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.pipeline import make_pipeline

# model=make_pipeline(TfidfVectorizer(),MultinomialNB())
# model.fit(train.data,train.target)

# with open('model.pkl','wb') as f:
#     pickle.dump(model,f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

def predict_category(s,train=train,model=model):
    pred=model.predict([s])
    print(pred)
    return train.target_names[pred[0]]

while True:
    askme = input('\nEnter news headline : ')
    predicted_news = predict_category(askme) 
    print(predicted_news)
