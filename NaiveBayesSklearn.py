import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix

import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

df = pd.read_csv('spam_ham_dataset.csv', engine= 'python')

def textClean(text):
    nopunc = ''.join([char for char in text if char not in string.punctuation])
    nopunc = set(nopunc.split())
    clean = ' '.join([word for word in nopunc if word.lower() not in stopwords.words('english')])
    return clean

X = df['text'].apply(textClean)
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=None)
vectorizer = CountVectorizer(ngram_range=(1,2)).fit(X_train)
X_train = vectorizer.transform(X_train)
classifier = MultinomialNB(alpha=0.1)
classifier.fit(X_train,y_train)
y_preds = classifier.predict(vectorizer.transform(X_test))
print('Confusion Matrix:\n',confusion_matrix(y_test,y_preds))
print('Accuracy:',accuracy_score(y_test,y_preds))