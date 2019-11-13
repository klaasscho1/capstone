import nltk
import time
import itertools
import json
import pickle
import truecase
import argparse
import _pickle as cPickle
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from nltk.corpus import wordnet
from string import punctuation
from collections import Counter
from math import log
from datetime import datetime
from stanford_nlp import StanfordNLP
from transformer import Transformer, TransformerStep
from cluster_metric_visualizers import ClusterScoreVisualizers
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
from whiten import whiten

print("Preparing argument parser...")

parser = argparse.ArgumentParser()
parser.add_argument('-c', action='store_true')  # Use cached everything
parser.add_argument('-t', action='store_true')  # Use cached TF-IDF matrix
parser.add_argument('-p', action='store_true')  # Use cached pre-processed data
parser.add_argument('-k', action='store_true')  # Use cached K-means clustering data
parser.add_argument('-s', action='store_true')  # Visualize silhouette score for k-clusters
parser.add_argument('-S', action='store_true')  # Save results after run
parser.add_argument('-e', action='store_true')  # Visualize elbow graph for k-clusters
parser.add_argument('-z', action='store_true')  # Use cached steps if available

options = parser.parse_args()

# UNCOMMENT TO DOWNLOAD NECESSARY NLTK-PACKAGES
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# nltk.download('wordnet')

start = time.time()

USE_CACHED_TFIDF = options.t or options.c
USE_CACHED_PREPROCESSING = USE_CACHED_TFIDF or options.p
USE_CACHED_K_MEANS = options.c or options.k
USE_CACHED_STEPS = options.c or options.z
SAVE_RESULTS = True

# File paths
PRE_PROCESSING_CACHE_PATH = "model_cache/no_ner_6nov/preprocessing_cache.pkl"
TF_IDF_CACHE_PATH = "model_cache/no_ner_6nov/tfidf_cache.pkl"
K_MEANS_CACHE_PATH = "model_cache/no_ner_6nov/k_means_cache.pkl"
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


def _step(name, transformation, keys=None):
    if keys is None:
        keys = ["title", "lead", "body"]
    return TransformerStep(name=name, transformer=transformation, keys=keys)


def get_paragraphs(text: str) -> [str]:
    return [p for p in text.split("\n\n") if p.strip() != ""]


def index_or_default(arr, ind, default):
    try:
        return arr[ind]
    except IndexError:
        return default


def load_articles():
    with open(DATA_PATH) as json_file:
        raw_data = json.load(json_file)

    _docs = [{"original_article": _article,
              "title": _article["content"]["title"],
              "publication": _article["mfc_annotation"]["source"],
              "year": _article["mfc_annotation"]["year"],
              "lead": get_paragraphs(_article["content"]["body"])[0]
              if len(get_paragraphs(_article["content"]["body"])) >= 1 else "",
              "body": " ".join(get_paragraphs(_article["content"]["body"])[1:])
              if len(get_paragraphs(_article["content"]["body"])) > 1 else ""}
             for _article in raw_data
             if _article["status"] == "VALID"
             and not _article["mfc_annotation"]["irrelevant"]]

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


# Removes tokens with only punctuation characters from a list of tokens
def remove_punctuation_tokens(pos_tokens: [(str, str)]) -> [(str, str)]:
    return [(s, pos) for (s, pos) in pos_tokens
            if not all(c in punctuation for c in s)]


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


def pre_process_data(data=None):
    if data is None:
        data = load_articles()
    print("Starting pre-processing:")

    start_pp = time.time()

    _prepped_data = Transformer.transform(data, transforming_steps)

    time_pp = str(round(time.time() - start_pp, 2))

    print("Finished pre-processing in {} sec!".format(time_pp))

    return _prepped_data


def get_word_frequency(doc):
    counter = Counter()

    # Count words from title double
    for (key, weight) in [("title", 1), ("lead", 1), ("body", 1)]:
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
    max_docs = round(0.5 * no_docs)

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


cache_cache = {}


def _cache_step(name, step, cache_id):
    step_cache_filename = "model_cache/steps/step_{}.pkl".format(cache_id)

    if cache_id in cache_cache:
        step_cache = cache_cache[cache_id]
    else:
        try:
            with open(step_cache_filename, "rb") as step_cache_file:
                step_cache = pickle.load(step_cache_file)
            cache_cache[cache_id] = step_cache
        except:
            step_cache = {}
            cache_cache[cache_id] = step_cache

    def caching_step(_input):
        result = step(_input)

        if isinstance(_input, list):
            input_key = tuple(_input)
        else:
            input_key = _input

        step_cache[input_key] = result
        cache_cache[cache_id] = step_cache
        return result

    if USE_CACHED_STEPS:
        def cached_step(_input):
            if isinstance(_input, list):
                input_key = tuple(_input)
            else:
                input_key = _input

            if input_key in step_cache:
                cached_result = step_cache[input_key]
                return cached_result
            else:
                print("No cached result for input, calculating result.")
                return caching_step(_input)

        return _step(name + " (from cache)", cached_step)
    else:
        return _step(name + " (caching)", caching_step)


transforming_steps = [
    _step("Restore true case of titles", truecase.get_true_case, ["title"]),
    #_step("Tokenize", nltk.word_tokenize),
    _cache_step("Filter named-entities (Stanford)", remove_named_entities_stf, cache_id="ner"),
    _cache_step("POS-tag", nltk.pos_tag, cache_id="pos"),
    _step("Strip punctuation", remove_punctuation_tokens),
    _step("Convert POS-tags to WordNet", convert_pos_tags_wordnet),
    _step("Lowercase", lambda sl: [(st.lower(), pos) for (st, pos) in sl]),
    _step("Filter only nouns, adjectives, adverbs", filter_pos),
    _step("Remove stopwords", remove_stopwords),
    _step("Remove numbers", remove_numbers),
    _step("Lemmatize", lambda word_pos: [(lemmatizer.lemmatize(word, pos), pos) for (word, pos) in word_pos]),
    _step("Merge tokens and POS", lambda word_pos: ["{}-{}".format(token, pos) for (token, pos) in word_pos]),
]

# Load pre-processed data from cache, or process it first
if USE_CACHED_PREPROCESSING:
    print("Loading cached preprocessed docs...")
    with open(PRE_PROCESSING_CACHE_PATH, 'rb') as pp_cache_file:
        prepped_data = pickle.load(pp_cache_file)
else:
    prepped_data = pre_process_data(data=load_articles())

    #print("Removing words not in title or lead from body")

    #for index, article in enumerate(prepped_data):
    #    prepped_data[index]["body"] = [pos_token for pos_token in article["body"]
    #                                   if pos_token in article["title"]
    #                                   or pos_token in article["lead"]]

    for key in cache_cache.keys():
        step_cache_filename = "model_cache/steps/step_{}.pkl".format(key)
        with open(step_cache_filename, "wb") as step_cache_file:
            pickle.dump(cache_cache[key], step_cache_file)

    # Save pre-processing result to cache
    with open(PRE_PROCESSING_CACHE_PATH, 'wb') as pp_cache_file:
        pickle.dump(prepped_data, pp_cache_file)

if USE_CACHED_TFIDF:
    print("Loading cached TF-IDF matrix...")
    with open(TF_IDF_CACHE_PATH, 'rb') as tfidf_cache_file:
        (words, tf_idf_mat) = pickle.load(tfidf_cache_file)
else:
    print("Calculating word and document frequencies...")

    doc_frequencies = [get_word_frequency(doc) for doc in prepped_data]
    doc_frequencies = filter_doc_frequencies(doc_frequencies)
    total_frequencies = get_doc_frequencies(doc_frequencies)

    words = list(total_frequencies.keys())

    print("Number of relevant words in corpus: {}".format(len(total_frequencies)))
    print("Most common 10: ", total_frequencies.most_common(10))

    start_tfidf = time.time()

    print("Calculating TF-IDF values for words in documents...")
    tf_idf_mat = [tf_idf_matrix_entry(word_frequency, total_frequencies, len(doc_frequencies)) for word_frequency in
                  doc_frequencies]
    tf_idf_mat = np.matrix(tf_idf_mat)

    # Save TF-IDF matrix to cache
    with open(TF_IDF_CACHE_PATH, 'wb') as tfidf_cache_file:
        pickle.dump((words, tf_idf_mat), tfidf_cache_file)

    time_tfidf = str(round(time.time() - start_tfidf, 2))
    print("Finished TF-IDF vectorization in {} sec.".format(time_tfidf))

# Perform normalization and whitening
print("Normalizing data...")
tf_idf_norm = normalize(tf_idf_mat, norm="l2")
# tf_idf_whi = whiten(tf_idf_norm)

X = tf_idf_norm

if options.e:
    print("Visualizing K-means elbow...")
    ClusterScoreVisualizers.k_means_elbow_graph(data=X, from_k=1, to_k=20)

if options.s:
    print("Visualizing Silhouette-score graph...")
    ClusterScoreVisualizers.sillhouette_coefficient(data=X, from_k=2, to_k=20)

k = 10

if USE_CACHED_K_MEANS:
    print("Loading cached k-means matrix...")
    with open(K_MEANS_CACHE_PATH, 'rb') as k_means_cache_file:
        k_means = pickle.load(k_means_cache_file)
else:
    print("Performing k-means clustering with k={} clusters...".format(k))

    k_means = KMeans(n_clusters=k)
    k_means.fit(X)

    # Save k-means clustering to cache
    with open(K_MEANS_CACHE_PATH, 'wb') as k_means_cache_file:
        pickle.dump(k_means, k_means_cache_file)

y_k_means = k_means.predict(X)

cluster_sizes = Counter(y_k_means)

print("Main terms per cluster")
order_centroids = k_means.cluster_centers_.argsort()[:, ::-1]
words_per_cluster = 5 # 1,2,3 stable?
cluster_words = []

for i in range(k):
    print("")
    print("Cluster {} (N={}):".format(i, cluster_sizes[i]))
    cluster_word_arr = []

    for ind in order_centroids[i, :words_per_cluster]:
        cluster_word_arr.append(words[ind])
        print(words[ind])

    cluster_words.append(cluster_word_arr)

# PCA distance is preserved
# MANIFOLD MDS
# Framing definition
# Whitening?

clustered_data = prepped_data.copy()

for i, cluster in enumerate(y_k_means):
    clustered_data[i].update(cluster_n=cluster)

if SAVE_RESULTS:
    now = datetime.now()
    timestamp = now.strftime("%m%d%Y_%H-%M-%S")
    print("Saving timestamped results (timestamp={})...".format(timestamp))

    with open("results/clusters{}.pkl".format(timestamp), 'wb') as clustered_data_file:
        pickle.dump(clustered_data, clustered_data_file)

    with open("results/kmeans{}.pkl".format(timestamp), 'wb') as kmeans_file:
        pickle.dump({
            "model": k_means,
            "k": k,
            "X": X,
            "y": y_k_means
        }, kmeans_file)

print("Finished!")

# Play with model, systematically (keep note of changes)
# Decide on the ones to change (no. clusters, doubling weight, )
# See which clusters 'survive'
# Record the results (titles, centroid distances, etc.)
# 1 week for modelling
