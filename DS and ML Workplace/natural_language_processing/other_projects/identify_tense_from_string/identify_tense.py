"""
Here’s a list of the tags, what they mean, and some examples:
    CC coordinating conjunction
    CD cardinal digit
    DT determiner
    EX existential there (like: “there is” … think of it like “there exists”)
    FW foreign word
    IN preposition/subordinating conjunction
    JJ adjective ‘big’
    JJR adjective, comparative ‘bigger’
    JJS adjective, superlative ‘biggest’
    LS list marker 1)
    MD modal could, will
    NN noun, singular ‘desk’
    NNS noun plural ‘desks’
    NNP proper noun, singular ‘Harrison’
    NNPS proper noun, plural ‘Americans’
    PDT predeterminer ‘all the kids’
    POS possessive ending parent‘s
    PRP personal pronoun I, he, she
    PRP$ possessive pronoun my, his, hers
    RB adverb very, silently,
    RBR adverb, comparative better
    RBS adverb, superlative best
    RP particle give up
    TO to go ‘to‘ the store.
    UH interjection errrrrrrrm
    VB verb, base form take
    VBD verb, past tense took
    VBG verb, gerund/present participle taking
    VBN verb, past participle taken
    VBP verb, sing. present, non-3d take
    VBZ  takes
    WDT wh-determiner which
    WP wh-pronoun who, what
    WP$ possessive wh-pronoun whose
    WRB wh-abverb where, when
"""
from nltk import word_tokenize, pos_tag
import numpy as np
import pandas as pd
# import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

default_text = 'I am reading and writing. my home is in India'
# default_text = 'book my flight, I will read this book'
# default_text = 'I am reading the book, and tommorrow i am going to book the tickets'

my_dict = {
    'CC' : 'coordinating conjunction',
    'CD' : 'cardinal digit',
    'DT' : 'determiner',
    'EX' : 'existential there',
    'FW' : 'foreign word',
    'IN' : 'preposition/subordinating conjunction',
    'JJ' : 'adjective',
    'JJR' : 'adjective, comparative',
    'JJS' : 'adjective, superlative',
    'LS' : 'list marker',
    'MD' : 'modal (ex - could, will)',
    'NN' : 'noun, singular',
    'NNS' : 'noun, plural ',
    'NNP' : 'proper noun, singular',
    'NNPS' : 'proper noun, plural',
    'PDT' : 'predeterminer',
    'POS' : 'possessive ending',
    'PRP' : 'personal pronoun',
    'PRP$' : 'possessive pronoun',
    'RB' : 'adverb',
    'RBR' : 'adverb, comparative',
    'RBS' : 'adverb, superlative',
    'RP' : 'particle',
    'TO' : 'to',
    'UH' : 'interjection',
    'VB' : 'verb, base form',
    'VBD' : 'verb, past tense',
    'VBG' : 'verb, present participle',
    'VBN' : 'verb, past participle',
    'VBP' : 'verb, present',
    'VBP' : 'verb, 3rd person present',
    'VBZ' : 'verb simple past',
    'WDT' : 'wh-determiner',
    'WP' : 'wh-pronoun',
    'WP$' : 'possessive wh-pronoun',
    'WRB' : 'wh-abverb',
}


text = input('Enter a sentence (Enter to use default) :')
if(text == ''):
    text = default_text

tokens = word_tokenize(text)
tags = pos_tag(tokens)

words = []
tenses = []

for item in tags:
    if item[1] in my_dict.keys():
        # print(f'{item[0]} - {my_dict[item[1]]}')
        word = item[0]
        tense = my_dict[item[1]]
    else:
        # print(f'{item[0]} - {item[1]}')
        word = item[0]
        tense = item[1]
    words.append(word)
    tenses.append(tense)

print('\nWords with their tense :\n')
df = pd.DataFrame({'Word':words, 'Tense':tenses})
print(df)
