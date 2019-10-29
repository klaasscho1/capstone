import nltk
import time
import itertools
import json
import pickle
import truecase
import argparse
import _pickle as cPickle
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from nltk.corpus import wordnet
from string import punctuation
from collections import Counter
from math import log
from stanford_nlp import StanfordNLP
from transformer import Transformer, TransformerStep
from cluster_metric_visualizers import ClusterScoreVisualizers
import truecaser.Truecaser as tc
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
from nltk.tag import StanfordNERTagger

parser = argparse.ArgumentParser()
parser.add_argument('-t', action='store_true')
parser.add_argument('-p', action='store_true')

options = parser.parse_args()

# UNCOMMENT TO DOWNLOAD NECESSARY NLTK-PACKAGES
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# nltk.download('wordnet')

start = time.time()

USE_CACHED_TFIDF = options.t
USE_CACHED_PREPROCESSING = USE_CACHED_TFIDF or options.p

# File paths
PRE_PROCESSING_CACHE_PATH = "model_cache/preprocessing_cache.pkl"
TF_IDF_CACHE_PATH = "model_cache/tfidf_cache.pkl"
DATA_PATH = 'data.json'

# Storing WordNet POS-tags locally improves performance
ADJ = wordnet.ADJ
VERB = wordnet.VERB
NOUN = wordnet.NOUN
ADV = wordnet.ADV

print("Connecting to Stanford CoreNLP server...")
sf_nlp = StanfordNLP()

lemmatizer = WordNetLemmatizer()
tokenizer = TreebankWordTokenizer()
detokenizer = TreebankWordDetokenizer()


def _step(name, transformation, keys=["title", "body"]):
    return TransformerStep(name=name, transformer=transformation, keys=keys)


def load_articles():
    with open(DATA_PATH) as json_file:
        raw_data = json.load(json_file)

    _docs = [{"title": article["content"]["title"],
              "publication": article["mfc_annotation"]["source"],
              "year": article["mfc_annotation"]["year"],
              "body": article["content"]["body"]}
             for article in raw_data
             if article["status"] == "VALID"
             and not article["mfc_annotation"]["irrelevant"]]

    return _docs


# Converts TreeBank POS-tag to WordNet tag
def get_wordnet_pos(treebank_tag: str) -> str:
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
def strip_punctuation(s: str) -> str:
    return ''.join(c for c in s if c not in punctuation)


# Removes punctuation characters from a list of tokens
def remove_punctuation_tokens(pos_tokens: [(str, str)]) -> [(str, str)]:
    return [(s, pos) for (s, pos) in pos_tokens if s not in punctuation]


# Converts all TreeBank POS-tags in a list to WordNet tags
def convert_pos_tags_wordnet(pos_tagged_words: [(str, str)]) -> [(str, str)]:
    wn_tagged_words = []

    for (word, pos) in pos_tagged_words:
        wn_pos = get_wordnet_pos(pos)
        wn_tagged_words.append((word, wn_pos))

    return wn_tagged_words


# Filters out all tokens that are not POS-tagged as a noun, adjective, or adverb
def filter_pos(sl: (str, str)) -> (str, str):
    return [(st, pos) for (st, pos) in sl if pos in [NOUN, ADJ, ADV]]


# Removes unnecessary stopwords like 'on,' 'with,' etc.
def remove_stopwords(words: [(str, str)]) -> [(str, str)]:
    _stopwords = set(stopwords.words('english'))
    words_filtered = []

    for (word, pos) in words:
        if word not in _stopwords:
            words_filtered.append((word, pos))

    return words_filtered


# Removes numbers from a list of tokens
def remove_numbers(tokens: [(str, str)]) -> [(str, str)]:
    return [(token, pos) for (token, pos) in tokens if not token.isdigit()]


# Removes named entities like names of cities and enterprises from a list
def remove_named_entities(pos_tagged_words: [(str, str)]) -> [(str, str)]:
    chunked = nltk.ne_chunk(pos_tagged_words)
    tokens = [leaf for leaf in chunked if type(leaf) != nltk.Tree]
    return tokens


# Remove every word that is referring to a person, organization or country
def remove_named_entities_stf(s: str) -> [str]:
    tagged_strs = sf_nlp.ner(s)
    return [s for (s, tag) in tagged_strs if tag not in ["PERSON", "ORGANIZATION", "COUNTRY"]]


def pre_process_data(data=load_articles()):
    print("Starting pre-processing:")

    start_pp = time.time()

    prepped_data = Transformer.transform(data, transforming_steps)

    time_pp = str(round(time.time() - start_pp, 2))

    print("Finished pre-processing in {} sec!".format(time_pp))

    return prepped_data


def get_word_frequency(doc):
    counter = Counter()

    # Count words from title double
    for (key, weight) in [("title", 2), ("body", 1)]:
        for _ in itertools.repeat(None, weight):
            counter.update(doc[key])

    return counter


def get_doc_frequencies(word_frequencies):
    counter = Counter()

    for document_frequencies in word_frequencies:
        counter += document_frequencies

    return counter


def filter_doc_frequencies(document_frequencies):
    in_no_docs = Counter()

    for doc in document_frequencies:
        for word in doc:
            if doc[word] > 0:
                in_no_docs[word] += 1

    no_docs = len(document_frequencies)

    # Remove words that appear in less than 5,
    # or more than 40% of articles
    min_docs = 5
    max_docs = round(0.4 * no_docs)

    filtered_words = []

    for word in in_no_docs:
        if min_docs <= in_no_docs[word] < max_docs:
            filtered_words.append(word)

    f_doc_frequencies = []

    for doc in document_frequencies:
        filtered_doc_frequency = Counter()
        for word in filtered_words:
            if word in doc:
                filtered_doc_frequency[word] = doc[word]
        f_doc_frequencies.append(filtered_doc_frequency)

    return f_doc_frequencies


def tf_idf(doc_frequency, _total_frequencies, N):
    tf_idf_for_word = {}

    for word in doc_frequency:
        tf = doc_frequency[word]
        df = _total_frequencies[word]

        tf_idf_for_word[word] = tf * log(N / float(df))

    return tf_idf_for_word


def tf_idf_matrix_entry(doc_frequency, _total_frequencies, N):
    doc_tf_idf = []

    for word in _total_frequencies:
        if word in doc_frequency:
            tf = doc_frequency[word]
        else:
            tf = 0

        df = _total_frequencies[word]

        if df > 0:
            doc_tf_idf.append(tf * log(N / float(df)))
        else:
            doc_tf_idf.append(0)

    return doc_tf_idf


transforming_steps = [
    _step("Restore true case of titles", truecase.get_true_case, ["title"]),
    _step("Filter named-entities (Stanford)", remove_named_entities_stf),
    _step("POS-tag", nltk.pos_tag),
    _step("Strip punctuation", remove_punctuation_tokens),
    _step("Convert POS-tags to WordNet", convert_pos_tags_wordnet),
    _step("Lowercase", lambda sl: [(st.lower(), pos) for (st, pos) in sl]),
    _step("Filter only nouns, adjectives, adverbs", filter_pos),
    _step("Remove stopwords", remove_stopwords),
    _step("Remove numbers", remove_numbers),
    _step("Lemmatize", lambda word_pos: [lemmatizer.lemmatize(word, pos) for (word, pos) in word_pos])
]

if USE_CACHED_TFIDF:
    print("Loading cached TF-IDF matrix...")
    with open(TF_IDF_CACHE_PATH, 'rb') as tfidf_cache_file:
        tf_idf_mat = pickle.load(tfidf_cache_file)
else:
    # Load pre-processed data from cache, or process it first
    if USE_CACHED_PREPROCESSING:
        print("Loading cached preprocessed docs...")
        with open(PRE_PROCESSING_CACHE_PATH, 'rb') as pp_cache_file:
            prepped_data = pickle.load(pp_cache_file)
    else:
        prepped_data = pre_process_data(data=load_articles())

        # Save pre-processing result to cache
        with open(PRE_PROCESSING_CACHE_PATH, 'wb') as pp_cache_file:
            pickle.dump(prepped_data, pp_cache_file)

    print("Calculating word and document frequencies...")

    doc_frequencies = [get_word_frequency(doc) for doc in prepped_data]
    doc_frequencies = filter_doc_frequencies(doc_frequencies)
    total_frequencies = get_doc_frequencies(doc_frequencies)

    print("Number of relevant words in corpus: {}".format(len(total_frequencies)))
    print("Most common 10: ", total_frequencies.most_common(10))

    start_tfidf = time.time()

    print("Calculating TF-IDF values for words in documents...")
    tf_idf_mat = [tf_idf_matrix_entry(word_frequency, total_frequencies, len(doc_frequencies)) for word_frequency in
                  doc_frequencies]
    tf_idf_mat = np.matrix(tf_idf_mat)

    # Save TF-IDF matrix to cache
    with open(TF_IDF_CACHE_PATH, 'wb') as tfidf_cache_file:
        pickle.dump(tf_idf_mat, tfidf_cache_file)

    time_tfidf = str(round(time.time() - start_tfidf, 2))
    print("Finished TF-IDF vectorization in {} sec.".format(time_tfidf))

# Perform l2 normalization
tf_idf_norm = normalize(tf_idf_mat, norm="l2")

X = tf_idf_norm

# UNCOMMENT TO SHOW K-MEANS ELBOW GRAPH
print("Visualizing K-means elbow...")
ClusterScoreVisualizers.k_means_elbow_graph(data=X, from_k=1, to_k=20)

# UNCOMMENT TO SHOW SILHOUETTE SCORE GRAPH
# print("Visualizing Silhouette-score graph...")
# ClusterScoreVisualizers.sillhouette_coefficient(data=X, from_k=2, to_k=20)

k = 8

# print("Performing k-means clustering with k={} clusters...".format(k))

# k_means = KMeans(n_clusters=k)
# y_k_means = k_means.fit_predict(X)

print("Finished!")
