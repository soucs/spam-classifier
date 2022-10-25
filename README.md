# Building Spam Classifier using Naive Bayes’ Algorithm from Scratch
Building Spam Classifier using Naive Bayes’ Algorithm
from Scratch
## I. Libraries Used
1. Numpy - For working with vectors.
2. Pandas - For manipulating the dataset
3. String - For the list of all punctuation. To remove all punctuation while data
preprocessing.
4. NLTK - For the list of stopwords in english. To remove all stopwords while data
preprocessing.
5. Decimal - Multiplying different probabilities give a value with a large number of decimal
places. Since Numpy cannot handle such numbers, we use the Decimal module to do
correctly-rounded floating point arithmetic.
## II. Given Dataset
Dataset of 5171 rows and 4 columns. The columns are:
1. Unnamed: 0 - Indexing of data by original source (int)
2. label - Labels of Emails which can be either Spam or Ham (string)
3. text - Entire email text (string)
4. label_num - labels encoded to 0 (ham) or 1 (ham) (int)
* The data has no empty cells and no duplicate rows.
* There are 1499 spam mails and 3672 ham mails, i.e. about 29% of the mails are spams
* We only need the ‘text’ and ‘label_num’ columns for building the classifier.
## III. Preprocessing (textClean(text) function)
* Removing punctuation from the mail texts.
* Removing stopwords (like ‘is’, ‘and’, ‘the’) from the mail texts.
## IV. Train-test Split
Splitting the dataset into train and test data in a 7:3 ratio (3620 train and 1551 test in our case).
X_train and y_train will be used to train the model, and X_test and y_test will be used to test how
well our model works. The data is shuffled to ensure randomness while doing this split.
## V. Vectorization
The string data should be converted to some quantitative form to train the model. This is achieved
through vectorizing the mail.
* Creating vocabulary - A list of all unique words in the train dataset is created. The length
of this list will serve as the dimension of each vector while vectorizing the emails (in one
case, it came out as 40067)
* For each email, a vector of dimension the same as the length of the vocabulary list is
created, with all its values initialized to zero. Following this, for each word in the
vocabulary list, we check if the word is present in the email. If it is, the vector value at the corresponding index is set as 1. Thus we get a vector to represent each email in the
train dataset. The same is also done for the test dataset.
* The y_train and y_test data are also converted into a vector format.
## VI. Model Training
The model is in the form of a number of probability values, using which we can calculate the
probability of a new email being spam or ham. This includes:
* prob_x_given_spam = A list containing the conditional probabilities of each word, given
spam, i.e. P(word/spam).
* prob_x_given_ham = A list containing the conditional probabilities of each word, given
ham, i.e. P(word/ham).
* prob_spam = Probability of spam = No.of spam emails/Total no.of emails in the dataset
* prob_ham = Probability of ham = 1 - Probability of spam

Zero-Frequency Problem - One of the disadvantages of Naive-Bayes is that if you have no
occurrences of a particular word in any spam/ham emails, then the conditional probability of that
word given spam/ham returns 0. This will get a zero when all the probabilities are multiplied. To
solve this, 1 is added to the count of every word for each class (spam/ham), so that they take a
small class-conditional probability value. This process of ‘smoothing’ our data by adding a
number is known as additive smoothing, also called Laplace smoothing. </br>
Here, we are adding a small probability (1018) to all frequency-based probability estimates which
has an equivalent effect.

## VII. Getting Predictions
For each item in X_test,
* Calculate the class_prior_spam, likelihood_spam, class_prior_ham, likelihood_ham and predictor_prior.
* Calculate the posterior probability for spam and ham (post_prob_spam and post_prob_ham).
* Decide as spam (1) or ham (0) based on which posterior probability is higher.

## VIII. Accuracy and Confusion Matrix
To know how well our model works we find the accuracy (true predictions by no.of test elements) and display the confusion matrix showing true positive (TP), true negative (TN), false positive (FP) and false negative (FN) counts. </br>
We test for both X_test and X_train to get test and train accuracy.
* Test accuracy comes out around 97%
* Train accuracy comes out around 99%

Training the model multiple times gives us accuracy values very near to 97% and 99% for test and train data respectively. This implies our model works well when generalized.
