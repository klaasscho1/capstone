import nltk
import time
import itertools
import json
import sys
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from string import punctuation
from collections import Counter
from math import log
from model.stanford_nlp import StanfordNLP
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import numpy as np
from model.cluster_metric_visualizers import ClusterScoreVisualizers

# UNCOMMENT TO DOWNLOAD NECESSARY NLTK-PACKAGES
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# nltk.download('wordnet')

start = time.time()

USE_CACHED_TFIDF = True
USE_CACHED_PREPROCESSING = USE_CACHED_TFIDF or True

# File paths
PRE_PROCESSING_CACHE_PATH = "model_cache/preprocessing_cache.pkl"
TF_IDF_CACHE_PATH = "model_cache/tfidf_cache.pkl"
DATA_PATH = './data.json'

# Storing WordNet POS-tags locally improves performance
ADJ = wordnet.ADJ
VERB = wordnet.VERB
NOUN = wordnet.NOUN
ADV = wordnet.ADV

print("Connecting to Stanford CoreNLP server...")
sf_nlp = StanfordNLP()

lemmatizer = WordNetLemmatizer()


def load_articles():
    with open(DATA_PATH) as json_file:
        raw_data = json.load(json_file)

    docs = [{"title": article["content"]["title"],
             "publication": article["mfc_annotation"]["source"],
             "year": article["mfc_annotation"]["year"],
             "body": article["content"]["body"]}
            for article in raw_data
            if article["status"] == "VALID"
            and not article["mfc_annotation"]["irrelevant"]]

    return docs

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


def filter_doc_frequencies(doc_frequencies):
    in_no_docs = Counter()

    for doc in doc_frequencies:
        for word in doc:
            if doc[word] > 0:
                in_no_docs[word] += 1

    no_docs = len(doc_frequencies)
    min_docs = 5
    max_docs = round(0.4 * no_docs) # 40% of documents

    filtered_words = []

    for word in in_no_docs:
        if in_no_docs[word] >= min_docs and in_no_docs[word] < max_docs:
            filtered_words.append(word)

    f_doc_frequencies = []

    for doc in doc_frequencies:
        filtered_doc_frequency = Counter()
        for word in filtered_words:
            if word in doc:
                filtered_doc_frequency[word] = doc[word]
        f_doc_frequencies.append(filtered_doc_frequency)

    return f_doc_frequencies


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

        if df > 0:
            doc_tf_idf.append(tf * log(N/float(df)))
        else:
            doc_tf_idf.append(0)

    return doc_tf_idf

pre_processing_steps = [
    ("Strip punctuation", strip_punctuation),
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

    for (name, transformer) in pre_processing_steps:
        print("-> " + name)

        new_data = []
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


if USE_CACHED_TFIDF:
    print("Loading cached TF-IDF matrix...")
    with open(TF_IDF_CACHE_PATH, 'rb') as tfidf_cache_file:
        tf_idf_mat = pickle.load(tfidf_cache_file)
else:
    if USE_CACHED_PREPROCESSING:
        print("Loading cached preprocessed docs...")
        with open(PRE_PROCESSING_CACHE_PATH, 'rb') as pp_cache_file:
            prepped_docs = pickle.load(pp_cache_file)
    else:
        print("Loading dataset...")

        docs = load_articles()

        print("Loaded {} articles!".format(len(docs)))

        print("Starting text-cleanup:")

        start_pp = time.time()

        prepped_docs = prepare(docs)

        time_pp = str(round(time.time() - start_pp, 2))

        print("Finished text-cleanup in {} sec!".format(time_pp))

        # Save pre-processing result to cache
        with open(PRE_PROCESSING_CACHE_PATH, 'wb') as pp_cache_file:
            pickle.dump(prepped_docs, pp_cache_file)

    print("Calculating word and document frequencies...")

    doc_frequencies = [get_word_frequency(doc) for doc in prepped_docs]
    doc_frequencies = filter_doc_frequencies(doc_frequencies)
    total_frequencies = get_doc_frequencies(doc_frequencies)

    print("Number of relevant words in corpus: {}".format(len(total_frequencies)))
    print("Most common 10: ", total_frequencies.most_common(10))

    start_tfidf = time.time()

    print("Calculating TF-IDF values for words in documents...")
    tf_idf_mat = [tf_idf_matrix_entry(word_frequency, total_frequencies, len(doc_frequencies)) for word_frequency in doc_frequencies]
    tf_idf_mat = np.matrix(tf_idf_mat)

    # Save TF-IDF matrix to cache
    with open(TF_IDF_CACHE_PATH, 'wb') as tfidf_cache_file:
        pickle.dump(tf_idf_mat, tfidf_cache_file)

    time_tfidf = str(round(time.time() - start_tfidf, 2))
    print("Finished TF-IDF vectorization in {} sec.".format(time_tfidf))

# Perform l2 normalization
tf_idf_norm = normalize(tf_idf_mat, norm="l1")

X = tf_idf_norm

# UNCOMMENT TO SHOW K-MEANS ELBOW GRAPH
# print("Visualizing K-means elbow...")
# ClusterScoreVisualizers.k_means_elbow_graph(data=X, from_k=1, to_k=20)

# UNCOMMENT TO SHOW SILLHOUETTE SCORE GRAPH
print("Visualizing Silhouette-score graph...")
ClusterScoreVisualizers.sillhouette_coefficient(data=X, from_k=2, to_k=20)

k = 13

print("Performing k-means clustering with k={} clusters...".format(k))

k_means = KMeans(n_clusters=k)
y_k_means = k_means.fit_predict(X)
