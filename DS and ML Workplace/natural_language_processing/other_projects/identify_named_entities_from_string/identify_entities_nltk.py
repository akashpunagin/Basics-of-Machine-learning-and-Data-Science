import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import conlltags2tree, tree2conlltags, ne_chunk
from pprint import pprint

# After installing nltk package
# nltk.download('maxent_ne_chunker')
# nltk.download('words')

# Example sentences
ex = 'European authorities fined Google a record $5.1 billion on Wednesday \
for abusing its power in the mobile phone market and ordered the company to alter its practices'
# ex = 'A spokeswoman for the F.B.I. did not respond to a message seeking comment about why Mr. Strzok \
# was dismissed rather than demoted. Firing Mr. Strzok, however, removes a favorite target of Mr. Trump from the ranks of the F.B.I. \
# and gives Mr. Bowdich and the F.B.I. director, Christopher A. Wray, a chance to move beyond the president’s ire.'

# Function to return tags (tense)
def sentence_to_tags(sentence):
    sentence = nltk.word_tokenize(sentence)
    sentence = nltk.pos_tag(sentence)
    return sentence

tagged_sentence = sentence_to_tags(ex)
print('Words of the sentence with tense :\n',tagged_sentence)

# Chunking - a process of extracting phrases from unstructured text

# Our chunk pattern consists of one rule, that a noun phrase, NP,
# should be formed whenever the chunker finds an optional determiner, DT,
# followed by any number of adjectives, JJ, and then a noun, NN.
pattern = 'NP: {<DT>?<JJ>*<NN>}'

# Create a chunk parser
cp = nltk.RegexpParser(pattern)
chunked_sentence = cp.parse(tagged_sentence)
# chunked_sentence.draw() # to display tree
print('\nThe chunked sentence (with tense) is:\n', chunked_sentence) # S at the first is denoting sentence

# tree2conlltags - returns a list of 3-tuples containing (word, tag, IOB-tag)
iob_tagged = tree2conlltags(chunked_sentence)
print('\nIOB tagged sentence:\n')
pprint(iob_tagged)

# nltk.ne_chunk() function will recognize named entities using a classifier, the classifier adds category labels such as PERSON, ORGANIZATION, and GPE.
# nltk.ne_chunk() is a named entity chunker to chunk the given list of tagged tokens
ne_tree = ne_chunk(tagged_sentence)
print('The chunked Sentence (with tense and entity) is :\n',ne_tree)

# The entities identified by ne_chunk
entities = {}
for chunk in ne_tree:
    if type(chunk) is nltk.Tree:
        t = ''.join(c[0] for c in chunk.leaves())
        entities[t] = chunk.label()
print('The entities of the sentence (identified by ne_chunk): ',entities)
