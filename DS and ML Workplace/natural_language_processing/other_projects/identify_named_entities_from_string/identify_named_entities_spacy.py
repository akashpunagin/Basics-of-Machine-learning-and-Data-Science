import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
from pprint import pprint
from path import Path
from spacy.tokens import Doc, Token, Span
from collections import Counter

# Example sentences
ex = 'European authorities fined Google a record $5.1 billion on Wednesday \
for abusing its power in the mobile phone market and ordered the company to alter its practices'
# ex = 'A spokeswoman for the F.B.I. did not respond to a message seeking comment about why Mr. Strzok \
# was dismissed rather than demoted. Firing Mr. Strzok, however, removes a favorite target of Mr. Trump from the ranks of the F.B.I. \
# and gives Mr. Bowdich and the F.B.I. director, Christopher A. Wray, a chance to move beyond the president’s ire.'

# en_core_web_sm - Assigns context-specific token vectors, POS tags, dependency parse and named entities
# load en_core_web_sm module into nlp
nlp = en_core_web_sm.load()

# Processing text with the nlp object returns a Doc object that holds all information about
# 1. the tokens
# 2. their linguistic features
# 3. their relationships
doc = nlp(ex)

# 1. Accessing token attributes
print('Tokens in the sentence :\n', [token.text for token in doc])

# 2. Linguistic features

# 2. a. Parts of Speech tags - POS tags
# Coarse-grained part-of-speech tags
print('\nCoarse grained POS tags :\n',[token.pos_ for token in doc])

# Token's lemma form with POS tag
pos_and_lemma = [(x.orth_,x.pos_, x.lemma_) for x in [token
                                      for token in doc
                                      if not token.is_stop and token.pos_ != 'PUNCT']]

print('\nToken(except stopwords), POS tag, lemma\n')
pprint(pos_and_lemma)


# Fine-grained part-of-speech tags
print('\nFine grained POS tags :\n',[token.tag_ for token in doc])

# 2. b. Syntactic dependencies
# Dependency labels
print('\nDependency labels :\n',[token.dep_ for token in doc])

# All entities
print('\nAll entities in the sentence :\n',doc.ents)

# 2. c. Named Entities
print('\nNamed Entities :\n', [(ent.text, ent.label_) for ent in doc.ents])

# Number of each entity
labels = [ent.label_ for ent in doc.ents]
print(f'\n{len(doc.ents)} entities is represented as {len(Counter(labels))} unique labels as follows:\n', Counter(labels))

# 2. d. Sentences (needs the dependency parser)
print('\nSentences :\n', [sent.text for sent in doc.sents])

# 2. e. Base noun phrases (needs the tagger and parser)
print('\nBase noun phrases :\n',[chunk.text for chunk in doc.noun_chunks])

# 3. Relationships
# Visualizing the relationships
displacy_dep_html = displacy.render(doc, jupyter=False, style='dep') # dep - dependency
displacy_ent_html = displacy.render(doc, jupyter=False, style='ent') # ent - entities

# Save as .svg file format
svg_dep = displacy_dep_html
output_path = Path("images/svg/displacy_dependency.svg")
output_path.open("w", encoding="utf-8").write(svg_dep)
svg_ent = displacy_ent_html
output_path = Path("images/svg/displacy_entity.svg")
output_path.open("w", encoding="utf-8").write(svg_ent)

# Save as HTML file
txt_file = open("images/html/displacy_dependency.html","w")
txt_file.write(displacy_dep_html)
txt_file.close()
txt_file = open("images/html/displacy_entity.html","w")
txt_file.write(displacy_ent_html)
txt_file.close()

# Method extensions (callable method)
# Register custom attribute on Doc class
def has_label(doc,label):
    is_found = False
    for ent in doc.ents:
        if ent.label_ == label:
            is_found = True
    return is_found
Doc.set_extension("has_label", method=has_label, force=True)

# Compute value of extension attribute with method
print('\nLabel in sentence? : ',doc._.has_label("GPE"))
