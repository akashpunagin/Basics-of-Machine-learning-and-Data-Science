import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer


# After installing nltk
# import nltk
# nltk.download('wordnet')
# nltk.download('stopwords')

# Get the data
messages = [line.rstrip() for line in open('smsspamcollection/SMSSpamCollection')]
print(len(messages))

print('1st 10 Messages:\n')
for message_no, message in enumerate(messages[:10]):
    print(message_no, message, '\n')

# Due to the spacing we can tell that this is a TSV ("tab separated values") file
# where the first column is a label saying whether the given message "ham" or "spam"
# The second column is the message itself

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

# Tokenizing the messages, (converting the message into words are we actually want)
# Check to make sure the function text_processing is working
print('\nNormalized Text (Removed Stopwords):\n',messages['message'].head(5).apply(text_processing))

# Vectorization
# we need to convert the list of tokens (also called lemmas) into vector form so that the machine learning algorithms can understand
# the three steps to do for bag-of-words model:
#     1. Count how many times does a word occur in each message (Known as term frequency)
#     2. Weigh the counts, so that frequent tokens get lower weight (inverse document frequency)
#     3 .Normalize the vectors to unit length, to abstract from the original text length

# CountVectorizer model will convert a collection of text documents to a matrix of token counts.

# Create instance of CountVectorizer and fit/train it with messages
print('Training CountVectorizer ...')
bow_transformer = CountVectorizer(analyzer=text_processing).fit(messages['message']) # bow (bag-of-words)

# Print total number of words in vocabulary
print('Total words in vocabulary of CountVectorizer model : ',len(bow_transformer.vocabulary_))

####################################################################################
# # Getting a message's bag-of-words vector
# message_4 = messages['message'][3]
# print('\n4th message : ',message_4)
#
# # The vector representation of 4th message
# bow_4 = bow_transformer.transform([message_4])
# print('Vector form of 4th message:\n',bow_4)
# # This means that there are seven unique words in message number 4 (after removing common stop words)
# # Two of them appear twice, the rest only once
# print('The words that appear twice are :')
# print(bow_transformer.get_feature_names()[4068])
# print(bow_transformer.get_feature_names()[9554])
######################################################################################

# transform the entire dataset
print('\nTransforming text documents to a matrix of token counts ...')
messages_bow = bow_transformer.transform(messages['message'])

print('\nShape of Sparse Matrix : ', messages_bow.shape)
print('Amount of Non-Zero occurences : ', messages_bow.nnz)

# Calculate sparsity, The number of zero-valued elements divided by the total number of elements is called the sparsity of the matrix
sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
print('Sparsity : {}'.format(sparsity))

# Now the term weighting and normalization can be done with TF-IDF (erm frequency-inverse document frequency), using TfidfTransformer
# TF-IDF is a statistical measure used to evaluate how important a word is to a document in a collection or corpus (a collection of texts)
# The importance increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the corpus

# Create instance of TfidfTransformer and fit with bow
print('\nTraining TfidfTransformer ...')
tfidf_transformer = TfidfTransformer().fit(messages_bow)

# check what is the IDF (inverse document frequency) of the word "u" and of word "university"
print('IDF of "u" : ',tfidf_transformer.idf_[bow_transformer.vocabulary_['u']])
print('IDF of "mail" : ',tfidf_transformer.idf_[bow_transformer.vocabulary_['mail']])

# Transform the entire bag-of-words corpus into TF-IDF corpus at once:
messages_tfidf = tfidf_transformer.transform(messages_bow)
print('messages_tfidf.shape : ',messages_tfidf.shape)

# Training a model using Naive Bayes classifier
X = messages_tfidf # features
y = messages['label'] # target
msg_train, msg_test, label_train, label_test = train_test_split(X, y, test_size=0.2)

# Training a model
spam_detection_model = MultinomialNB().fit(msg_train, label_train)

####################################################################################
# # Trying to classify a single message
# tfidf_4 = tfidf_transformer.transform(bow_4)
# print('Expected :', messages.label[3])
# print('Predicted :', spam_detection_model.predict(tfidf_4)[0])
####################################################################################

# Predicting all the messages from tfidf model
all_predictions = spam_detection_model.predict(msg_test)
print('All predictions : ',all_predictions)

# Model Evaluation
print('\nClassifiction Report:\n',classification_report(label_test,all_predictions))
