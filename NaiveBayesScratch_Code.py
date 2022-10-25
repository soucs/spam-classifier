import pandas as pd
import numpy as np
from decimal import Decimal

import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

df = pd.read_csv(r'/media/soucs/OS/Users/soumy/Documents/Python Notebooks/mi_sem_1/spam_ham_dataset.csv', engine= 'python')

# Data preprocessing
def textClean(text):
    nopunc = ''.join([char for char in text if char not in string.punctuation])
    nopunc = set(nopunc.split())
    clean = ' '.join([word for word in nopunc if word.lower() not in stopwords.words('english')])
    return clean
#applied_df = df.copy()
df['text'] = df['text'].apply(lambda mail:textClean(mail))

# Train-Test split (0.7/0.3)
test_df = df.sample(frac = 0.3)
train_df = pd.concat([df, test_df]).drop_duplicates(keep=False)

# Master list of unique words
big_string = ''
for text in train_df['text']:
  big_string = ' '.join([big_string,text])

big_string = big_string.split()
big_string = [*set(big_string)]

# Emails to vectors function
def word2vect(email, big_string=big_string):
  email = set(email.split())
  v = np.zeros(len(big_string))
  idx = [i for i,word in enumerate(big_string) if word in email]
  v[np.array(idx)] = 1
  return v

# Matrix of all vectors
X_train = np.array([word2vect(text) for text in train_df['text']])
X_test = np.array([word2vect(text) for text in test_df['text']])

y_train = np.array([n for n in train_df['label_num']])
y_test = np.array([n for n in test_df['label_num']])

# spam = 1; ham = 0
# Finding probabilities (genarating model using X_test)
def prob_x_given_y(y_output,word):
  #if type(y_output)==str: y_output = Encode(y_output)
  feature_given_output = X_train[:,word][np.where(y_train==y_output)] # Gives a particular word's column for a partucular y_output
  float_probab = sum(feature_given_output)/len(feature_given_output)
  return Decimal(float_probab)

prob_x_given_spam = [prob_x_given_y(1, word) for word in range(len(big_string))]
prob_x_given_ham = [prob_x_given_y(0, word) for word in range(len(big_string))]
prob_spam = sum(y_train)/len(y_train)
prob_ham = 1-prob_spam

# Function for model training and getting prediction
def NBModel(X):
  global prob_spam, prob_ham, prob_x_given_spam, prob_x_given_ham
  y_pred=[]
  for text in X:
    class_prior_spam = Decimal(prob_spam)
    likelihood_spam = np.prod([prob_x_given_spam[i]+Decimal(10**-18) for i,word in enumerate(text) if word==1])

    class_prior_ham = Decimal(prob_ham)
    likelihood_ham = np.prod([prob_x_given_ham[i]+Decimal(10**-18) for i,word in enumerate(text) if word==1])

    predictor_prior = likelihood_spam*class_prior_spam + likelihood_ham*class_prior_ham

    post_prob_spam = likelihood_spam * class_prior_spam / predictor_prior
    post_prob_ham = likelihood_ham * class_prior_ham / predictor_prior
    prediction = 1 if post_prob_spam>post_prob_ham else 0
    y_pred.append(prediction)
  return y_pred

def modelAccuracy(y_teach, y_pred):
    classes = np.unique(y_teach)
    confmat = np.zeros((len(classes), len(classes)))
    for i in range(len(classes)):
        for j in range(len(classes)):
           confmat[i, j] = np.sum((y_teach == classes[i]) & (y_pred== classes[j]))
    accuracy = (confmat[0,0]+confmat[1,1])/(confmat[0,0]+confmat[1,1]+confmat[0,1]+confmat[1,0])
    print("Confusion Matrix: \n", confmat)
    print("Accuracy: \n", accuracy)

# Getting predictions for train data and Printing accuaracy and confusion matrix
print('Train Data:')
y_pred = NBModel(X_train)
modelAccuracy(y_train, y_pred)

print()

# Getting predictions for test data and Printing accuaracy and confusion matrix
print('Test Data:')
y_pred = NBModel(X_test)
modelAccuracy(y_test, y_pred)