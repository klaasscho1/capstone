import nltk
import time
import pandas as pd
import numpy as np
import itertools
from collections import namedtuple
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from string import punctuation
from collections import Counter
from math import log

start = time.time()

#nltk.download('averaged_perceptron_tagger')
#nltk.download('maxent_ne_chunker')
#nltk.download('words')
#nltk.download('wordnet')

# Storing WordNet POS-tags locally improves performance
ADJ = wordnet.ADJ
VERB = wordnet.VERB
NOUN = wordnet.NOUN
ADV =  wordnet.ADV

lemmatizer = WordNetLemmatizer()

Article = namedtuple('Article', ['title', 'publication', 'date', 'body'])

demo_article = Article(
    title="Clinton Escalates Fight on Illegal Immigration",
    publication="The Washington Post",
    date="July 28, 1993",
    body="""
    President Clinton, surrounded by a bipartisan group of lawmakers whose states are struggling with illegal immigration, yesterday asked a receptive Congress for an additional $ 172.5 million and stronger law enforcement tools to fight illegal immigration.
    "We cannot and will not surrender our border to those who wish to exploit our history of compassion and justice," said Clinton, flanked by Vice President Gore and Attorney General Janet Reno, as he outlined the legislative details of proposals first announced in June. The plan was devised by a 12-agency task force directed by Gore, working with Congress.
    The key elements of the administration's bill include "expedited exclusion" hearings for immigrants claiming asylum as well as stiffer penalties for immigrant smugglers and prosecution of smugglers under racketeering laws. The plan also would provide for more asylum officers and up to 600 additional Border Patrol guards, and would improve the State Department's ability to identify potentially dangerous immigrants overseas to keep them from entering the country.
    The expedited exclusion hearings would amount to a new preliminary screening of aliens at entry points into the United States, with a full hearing granted only to applicants who prove they have a "credible fear" of persecution in their home countries. The exclusion rules would apply to the 15,000 refugees a year who request asylum when they are caught entering the country illegally. But they would not affect the estimated 300,000 immigrants who enter undetected each year, or the nearly 100,000 who apply for asylum without having been apprehended.
    The accelerated process is designed to take no more than five days, compared to the average 18 months a preliminary asylum hearing now takes, and was immediately denounced by civil liberties groups that said it may infringe immigrants' rights.
    Congressional reaction generally was more positive. Aides said that Sens. Edward M. Kennedy (D-Mass.) and Alan K. Simpson (R-Wyo.) are negotiating in hope of introducing the bill jointly. Rep. Charles E. Schumer (D-N.Y.), who has cosponsored a more far-reaching bill with Reps. Romano L. Mazzoli (D-Ky.) and Bill McCollum (R-Fla.), said the proposal "achieves a good balance between toughness and fairness." Mazzoli, chairman of the Judiciary subcommittee on international law, immigration and refugees, spoke favorably of the bill but said he might move to amend it slightly.
    Clinton's tough posture against illegal immigration represents a shift from his campaign, when he criticized restrictive GOP immigration policies and emphasized the nation's immigrant tradition. What has prompted the new effort, aides said, has been the highly publicized arrival of hundreds of Chinese migrants in the United States
    through organized smuggling rings and the links of an immigrant sheik and others of questionable immigration status to the World Trade Center bombing.
    "To treat terrorists and smugglers as immigrants dishonors the tradition of the immigrants who have made our nation great," Clinton said. "We must say no to illegal immigration so we can continue to say yes to legal immigration."
    But immigration advocates said it may be impossible to just say no to illegal migrants while still protecting bona fide refugees, and some members of Congress said they were wary of the plan. "We want to be absolutely certain that we do not compromise fair consideration of asylum requests for legitimate political refugees," said Rep. Jose E. Serrano (D-N.Y.), chairman of the Congressional Hispanic Caucus.
    By allowing an asylum officer to decide on the spot whether applicants deserve full-fledged hearings -- and by replacing several layers of judicial review with a single appeal to another immigration official -- the proposal would hurt refugees wary of talking to government officials just hours after arriving in the United States, critics assert.
    "The persons most hurt by this bill are those who are fleeing persecution. These are people who survived because they didn't share their confidences," said Carol Wolchok, director of the American Bar Association's Center for Immigration Law and Representation. "It's really a roulette as to whether you get an officer who listens to what you have to say and will give you an opportunity to go forward to prove your claim."
    Lucas Guttentag, director of the American Civil Liberties Union's Immigrants' Rights Project, agreed. Eliminating court appeals in order to speed up exclusion hearings "creates a veil of secrecy" around the Immigration and Naturalization Service that "just invites . . . arbitrary and discriminatory applications," he said.
    "This is not a system that's designed to make adequate decisions in a situation of life or death," Guttentag added. "The proposal is a badly misguided response by the president and some in Congress that panders to America's most primitive fears about immigration."
    A Gallup Poll in July found that 69 percent of respondents favored reducing immigration, which Gallup called its highest such finding since World War II.
    Civil liberties groups said they will lobby Congress for amendments to the bill to reduce the standard of proof and allow judicial review of asylum decisions.
    "The question shouldn't be can America wash its hands of these refugees, but can we offer a true refugee protection," said Warren Leiden, executive director of the American Immigration Lawyers Association. "I don't think America has to choose between efficiency and fairness."
    """
)._asdict()
docs = [demo_article.copy(), demo_article.copy()]

# Converts TreeBank POS-tag to WordNet tag
def get_wordnet_pos(treebank_tag):
    if treebank_tag[0] == 'J':
        return ADJ
    elif treebank_tag[0] == 'V':
        return VERB
    elif treebank_tag[0] == 'N':
        return NOUN
    elif treebank_tag[0] == 'R':
        return ADV
    else:
        return NOUN

# Removes punctuation characters
def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation)

# Converts all TreeBank POS-tags in a list to WordNet tags
def convert_pos_tags_wordnet(pos_tagged_words):
    wn_tagged_words = []

    for (word, pos) in pos_tagged_words:
        wn_pos = get_wordnet_pos(pos)
        wn_tagged_words.append((word, wn_pos))

    return wn_tagged_words

# Removes unnecessary stopwords like 'on,' 'with,' etc.
def remove_stopwords(words):
    _stopwords = set(stopwords.words('english'))
    words_filtered = []

    for (word, pos) in words:
        if word not in _stopwords:
            words_filtered.append((word, pos))

    return words_filtered

# Removes numbers from a list of tokens
def remove_numbers(tokens):
    return [(token, pos) for (token, pos) in tokens if not token.isdigit()]

# Removes named entities like names of cities and enterprises from a list
def remove_named_entities(pos_tagged_words):
    chunked = nltk.ne_chunk(pos_tagged_words)
    tokens = [leaf for leaf in chunked if type(leaf) != nltk.Tree]
    return tokens

def get_word_frequency(doc):
    counter = Counter()

    for (key, weight) in [("title", 2), ("body", 1)]:
        for _ in itertools.repeat(None, weight):
            counter.update(doc[key])

    return counter

def get_doc_frequencies(word_frequencies):
    counter = Counter()

    for doc in word_frequencies:
        counter.update(doc.keys())

    return counter

def doc_tf_idf(word_frequency, doc_frequencies, N):
    counter = Counter()

    for word in word_frequency:
        tf = word_frequency[word]
        df = doc_frequencies[word]

        counter[word] = tf * log(N/df)

    return counter


preprocessing_steps = [
    ("Strip punctuation", strip_punctuation),
    ("Tokenize", word_tokenize),
    ("POS-tag", nltk.pos_tag),
    ("Remove named-entities", remove_named_entities), # Disabled for performance reasons
    ("Convert POS-tags to WordNet", convert_pos_tags_wordnet),
    ("Lowercase", lambda sl: [(st.lower(), pos) for (st, pos) in sl]),
    ("Remove stopwords", remove_stopwords),
    ("Remove numbers", remove_numbers),
    ("Lemmatize", lambda word_pos: [(lemmatizer.lemmatize(word, pos), pos) for (word, pos) in word_pos])
]

def prepare(data):
    updating_data = data

    for (name, transformer) in preprocessing_steps:
        print("-> " + name)
        new_data = []
        start = time.time()
        for doc in updating_data:
            new_doc = doc
            for part in ["title", "body"]:
                new_doc[part] = transformer(doc[part])
            new_data.append(new_doc)
        end = time.time()
        print(".. " + str(round(end - start, 2)) + " sec")
        updating_data = new_data

    return new_data

print("Starting preparation of %i docs:" % len(docs))
start_pp = time.time()
prepped_docs = prepare(docs)
print("Finished preparation in " + str(round(time.time() - start_pp, 2)) + " sec!")

print("Calculating word and document frequencies.")
word_frequencies = [get_word_frequency(doc) for doc in prepped_docs]
doc_frequencies = get_doc_frequencies(word_frequencies)

print("Calculating TF-IDF values for words in documents.")
doc_tf_idfs = [doc_tf_idf(word_frequency, doc_frequencies, len(word_frequencies))
               for word_frequency in word_frequencies]

print("Success! Finished in " + str(round(time.time() - start, 2)) + " sec.")
