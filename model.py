import nltk
import time
import pandas as pd
import numpy as np
import itertools
import json
import sys
import pickle
from collections import namedtuple
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from string import punctuation
from collections import Counter
from math import log
from stanford_nlp import StanfordNLP
from sklearn.cluster import KMeans

start = time.time()

# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# nltk.download('wordnet')

# Storing WordNet POS-tags locally improves performance
ADJ = wordnet.ADJ
VERB = wordnet.VERB
NOUN = wordnet.NOUN
ADV = wordnet.ADV

USE_CACHED_PREPROCESSING = True

lemmatizer = WordNetLemmatizer()
sf_nlp = StanfordNLP()

DATA_PATH = './data.json'

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
    return(tokens)


def get_word_frequency(doc):
    counter = Counter()

    # Count words from title double
    for (key, weight) in [("title", 2), ("body", 1)]:
        for _ in itertools.repeat(None, weight):
            counter.update(doc[key])

    return counter


def get_doc_frequencies(word_frequencies):
    counter = Counter()

    for doc_frequencies in word_frequencies:
        counter += doc_frequencies

    return counter


def tf_idf(doc_frequency, total_frequencies, N):
    tf_idf_for_word = {}

    for word in doc_frequency:
        tf = doc_frequency[word]
        df = total_frequencies[word]

        tf_idf_for_word[word] = tf * log(N/float(df))

    return tf_idf_for_word


def tf_idf_matrix_entry(doc_frequency, total_frequencies, N):
    doc_tf_idf = []

    for word in total_frequencies:
        if word in doc_frequency:
            tf = doc_frequency[word]
        else:
            tf = 0

        df = total_frequencies[word]

        doc_tf_idf.append(tf * log(N/float(df)))

    return doc_tf_idf


preprocessing_steps = [
    ("POS-tag", sf_nlp.pos),
    # ("Remove named-entities", remove_named_entities),
    ("Convert POS-tags to WordNet", convert_pos_tags_wordnet),
    ("Lowercase", lambda sl: [(st.lower(), pos) for (st, pos) in sl]),
    ("Remove stopwords", remove_stopwords),
    ("Remove numbers", remove_numbers),
    ("Lemmatize", lambda word_pos: [lemmatizer.lemmatize(word, pos) for (word, pos) in word_pos])
]


class TransformerError(Exception):
    def __init__(self, exception):
        self.exception = exception
    pass


def prepare(data):
    updating_data = data

    for (name, transformer) in preprocessing_steps:
        print("-> " + name)

        new_data = []
        start = time.time()
        doc_cnt = 0

        for doc in updating_data:
            new_doc = doc
            print('Document {}/{}'
                  .format(doc_cnt, len(updating_data), str(round(time.time() - start))),
                  end='\r')

            try:
                for part in ["title", "body"]:
                    try:
                        new_doc[part] = transformer(doc[part])
                    except Exception as e:
                        raise TransformerError(exception=e)
            except TransformerError as e:
                doc_cnt += 1
                print("Cannot perform '{}' on document #{} because of exception:\n{}"
                      .format(name, doc_cnt, repr(e.exception)))
                continue

            doc_cnt += 1
            new_data.append(new_doc)

        end = time.time()
        sys.stdout.flush()
        updating_data = new_data

        print("Finished in " + str(round(end - start, 2)) + " sec\r")
        print("")

    return new_data


def prepare_sf(data):
    MODELS_DIR = './models/'
    stanfordnlp.download('en', MODELS_DIR)
    nlp = stanfordnlp.Pipeline(processors='tokenize,pos',
                               models_dir=MODELS_DIR,
                               treebank='en_ewt',
                               use_gpu=True,
                               pos_batch_size=3000)

    for doc in data:
        for part in ["title", "body"]:
            sf_doc = nlp(doc[part])


if USE_CACHED_PREPROCESSING:
    print("Loading cached preprocessed docs...")
    with open('preprocessing_cache.pkl', 'rb') as pp_cache_file:
        prepped_docs = pickle.load(pp_cache_file)
else:
    print("Loading dataset...")

    with open(DATA_PATH) as json_file:
        raw_data = json.load(json_file)

    docs = [{"title": article["content"]["title"],
             "publication": article["mfc_annotation"]["source"],
             "year": article["mfc_annotation"]["year"],
             "body": article["content"]["body"]}
            for article in raw_data
            if article["status"] == "VALID"
            and not article["mfc_annotation"]["irrelevant"]]

    print("Loaded {} articles!".format(len(docs)))

    print("Starting preparation of %i docs:" % len(docs))

    start_pp = time.time()
    prepped_docs = prepare(docs)
    time_pp = str(round(time.time() - start_pp, 2))

    print("Finished preparation in {} sec!".format(time_pp))

    with open('preprocessing_cache.pkl', 'wb') as pp_cache_file:
        pickle.dump(prepped_docs, pp_cache_file)

print("Calculating word and document frequencies.")
doc_frequencies = [get_word_frequency(doc) for doc in prepped_docs]
total_frequencies = get_doc_frequencies(doc_frequencies)

start_tfidf = time.time()

print("Calculating TF-IDF values for words in documents.")
tf_idf_mat = [tf_idf_matrix_entry(word_frequency, total_frequencies, len(doc_frequencies)) for word_frequency in doc_frequencies]

time_tfidf = str(round(time.time() - start_tfidf, 2))
print("Finished TF-IDF vectorization in {} sec.".format(time_tfidf))

print("Performing K-means clustering with n={} clusters".format(2))
kmeans = KMeans(n_clusters=5).fit(tf_idf_mat)

print("Success! Finished in {} sec.".format(str(round(time.time() - start, 2))))
