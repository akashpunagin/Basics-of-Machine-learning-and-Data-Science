import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer


# After installing nltk
# import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')

# Get the data using read_csv
messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t', names=["label", "message"])

print(messages.head())

# Exploratory Data Analysis
print(messages.describe())
print('\n')
print(messages.groupby('label').describe(),'\n')

# new column to detect how long the text messages are
messages['length'] = messages['message'].apply(len)
print(messages.head())

# Data Visualization

messages['length'].plot(bins=50, kind='hist')
plt.xlabel('Message Length')
# plt.show()

# To see if the message length is a distingushing feature between ham and spam messages
messages.hist(column='length', by='label', bins=50,figsize=(12,4))
# plt.show()

# Text Pre-processing

# Show some stop words
print('Some Stopwords are :', stopwords.words('english')[0:10])

# Function for Test Pre-processing
def text_processing(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    4. Stemming:  Stemming is a rudimentary rule-based process of stripping the suffixes (“ing”, “ly”, “es”, “s” etc) from a word.
    5. Lemmatization: Lemmatization, on the other hand, is an organized & step by step procedure of obtaining the root form of the word
    it makes use of vocabulary (dictionary importance of words) and morphological analysis (word structure and grammar relations)
    """
    # # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)

    # Now just remove any stopwords
    no_stopwords = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

    lemma_instance = WordNetLemmatizer()
    lemmas = [lemma_instance.lemmatize(word, "v") for word in no_stopwords]

    stem_instance = PorterStemmer()
    stems = [stem_instance.stem(word) for word in lemmas]

    return stems


# Creating the pipeline
print('\nCreating the pipeline ...')
pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_processing)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors with Naive Bayes classifier
])

# Train Test Split
X = messages['message'] # features
y = messages['label'] # target
msg_train, msg_test, label_train, label_test = train_test_split(X, y, test_size=0.3)

# Training Pipeline
print('\nTraining the Pipeline ...')
pipeline.fit(msg_train,label_train)

# Predictions
print('\nPredicting ...')
predictions = pipeline.predict(msg_test)

# Model Evaluation
print('\nClassifiction Report:\n',classification_report(label_test,predictions))
