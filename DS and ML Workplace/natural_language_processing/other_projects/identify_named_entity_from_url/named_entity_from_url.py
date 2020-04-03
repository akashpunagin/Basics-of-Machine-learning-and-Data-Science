from bs4 import BeautifulSoup
import requests
import re
from spacy import displacy
import en_core_web_sm
from pprint import pprint
from path import Path
from collections import Counter

# Function to convert url content to string
def url_to_string(url):
    res = requests.get(url)
    html = res.text
    soup = BeautifulSoup(html, 'html5lib')
    for script in soup(["script", "style", 'aside']):
        script.extract()
    return " ".join(re.split(r'[\n\t]+', soup.get_text()))

# url = 'https://www.nytimes.com/2018/08/13/us/politics/peter-strzok-fired-fbi.html?hp&action=click&pgtype=Homepage&clickSource=story-heading&module=first-column-region&region=top-news&WT.nav=top-news'
# url = 'https://www.nytimes.com/2020/04/02/us/politics/trump-russia-saudi-arabia-oil.html'
# url = 'https://www.nytimes.com/2020/04/02/us/politics/milwaukee-democratic-convention-delayed.html'
url = 'https://www.nytimes.com/2020/04/03/technology/coronavirus-masks-shortage.html'

# en_core_web_sm - Assigns context-specific token vectors, POS tags, dependency parse and named entities
# load en_core_web_sm module into nlp
nlp = en_core_web_sm.load()

url_content = url_to_string(url)

# Processing text with the nlp object returns a Doc object that holds all information about
# 1. the tokens
# 2. their linguistic features
# 3. their relationships
article = nlp(url_content)

n_entities = len(article.ents)
print('Total number of entities in url : ',n_entities)
print('\n1st 5 entities : ', article.ents[:5] , '...')


labels = [ent.label_ for ent in article.ents]
print('1st 5 Labels : ',labels[:5] , '...')
n_labels = len(Counter(labels))

# Named Entities
print('\nNamed Entities :\n')
pprint([(ent.text, ent.label_) for ent in article.ents][:10])
print(' ...\n ..')

print(f'\n{n_entities} entities is represented as {n_labels} unique labels as follows:\n', Counter(labels))

# Frequent tokens
frequent_tokens = [ent.text for ent in article.ents]
print('\nFrequent Tokens are : ',Counter(frequent_tokens).most_common(5))

# Visualizing
displacy_dep_html = displacy.render(article, jupyter=False, style='dep') # dep - dependency
displacy_ent_html = displacy.render(article, jupyter=False, style='ent') # ent - entities

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
