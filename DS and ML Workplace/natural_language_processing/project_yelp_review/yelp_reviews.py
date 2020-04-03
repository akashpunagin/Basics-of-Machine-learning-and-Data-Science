# Each observation in this dataset is a review of a particular business by a particular user.
#
# The "stars" column is the number of stars (1 through 5) assigned by the reviewer to the business
#
# The "cool" column is the number of "cool" votes this review received from other Yelp users.
# All reviews start with 0 "cool" votes, and there is no limit to how many "cool" votes a review can receive
# In other words, it is a rating of the review itself, not a rating of the business.
# The "useful" and "funny" columns are similar to the "cool" column.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix,classification_report
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize, pos_tag
sns.set_style('white')

# After installing nltk package
# import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')

# Get the data
yelp = pd.read_csv('yelp.csv')

print(yelp.head())
print(yelp.describe())
print(yelp.info())

# Create new column for number of words in text column called text length
yelp['text length'] = yelp['text'].apply(len)

print(yelp.head())

# Exploratory data analysis
# Use FacetGrid to create a grid of 5 histograms of text length based off of the star ratings
g = sns.FacetGrid(yelp,col='stars')
g.map(plt.hist,'text length')
# plt.show()

# Create a boxplot of text length for each star category
plt.figure()
sns.boxplot(x='stars',y='text length',data=yelp,palette='rainbow')
# plt.show()

# Create a countplot of the number of occurrences for each type of star rating.
plt.figure()
sns.countplot(x='stars',data=yelp,palette='rainbow')
# plt.show()

# Correlation between features grouped by 'stars column'
corr_df = yelp.groupby('stars').mean().corr()
print('Correlation Dataframe :\n',corr_df)

# Then use seaborn to create a heatmap
plt.figure()
sns.heatmap(corr_df,cmap='coolwarm',annot=True)
# plt.show()

# Specifying features and target
# yelp = yelp[(yelp.stars==1) | (yelp.stars==5)] # doing this will make things easier and improve the model
X = yelp['text'] # features
y = yelp['stars'] # target

# CountVectorizer model will convert a collection of text documents to a matrix of token counts.
# Create instance of CountVectorizer
cv = CountVectorizer()

# fit/train it with features, and assign transformed data back to X
print('\nTraining CountVectorizer ...')
X = cv.fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Training a Model
# Create instance of MultinomialNB
nb = MultinomialNB()

# fit/train the mdoel
nb.fit(X_train,y_train)

# Predictions
predictions = nb.predict(X_test)

# Model evaluation
print('Using Bag of words model and MultinomialNB')
print('Classifiction Report:\n',classification_report(y_test,predictions))
# print('\nConfusion Matrix:\n',confusion_matrix(y_test,predictions))

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
    6. Part of speech tagging
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

    # tokens = word_tokenize(stems)
    tags = pos_tag(stems)

    return tags


# Using Text Processing
# Creating a pipeline
pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_processing)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', RandomForestClassifier()),  # train on TF-IDF vectors with RandomForestClassifier (or MultinomialNB)
])

# Overwrite X and y, and use train_test_split
X = yelp['text'] # features
y = yelp['stars'] # target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Training Pipeline
print('\nTraining Pipeline ...')
pipeline.fit(X_train,y_train)

# Predictions
print('\nPredicting ...')
predictions = pipeline.predict(X_test)

# Model evaluation
print('\nUsing TfidfTransformer model and RandomForestClassifier')
print('Classifiction Report:\n',classification_report(y_test,predictions))
# print('\nConfusion Matrix:\n',confusion_matrix(y_test,predictions))
