from __future__ import print_function
from nltk.corpus import stopwords as sw
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer
import nltk
import json
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans, MiniBatchKMeans
import logging
from optparse import OptionParser
import sys
from time import time


def cluster_poems_per_text(data, clusters_number, seed):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    # number of clusters to find
    true_k = clusters_number

    t0 = time()

    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2)
    X = vectorizer.fit_transform(data)

    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=20, n_init=1, random_state=seed)
    km.fit(X)

    order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    terms = vectorizer.get_feature_names()
    dict_terms = {index: [] for index in range(true_k)}

    for i in range(true_k):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
            dict_terms[i].append(terms[ind])
        print()

    return km.labels_, dict_terms, km.inertia_, km.cluster_centers_