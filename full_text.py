#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

""""""

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from gensim.models import Word2Vec
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
import numpy as np
import json
import re
import os

from shift_detection import SklearnDataShiftDetector
from settings import get_resource_path
from pipelines import BasePipeline
from models import RandomForest


class HashingPipeline(BasePipeline):
    def __init__(self, estimator):
        self.pipe = Pipeline([('hashing vectorizer', HashingVectorizer()),
                              ('estimator', estimator)])


class TfIdfPipeline(BasePipeline):
    def __init__(self, estimator):
        self.pipe = Pipeline([('tf-idf vectorizer', TfidfVectorizer()),
                              ('estimator', estimator)])


class FullTextTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        """Helpers"""
        """LaTeX numbers and math formulas removal - $.+$"""
        self.numbers = re.compile(r"[\d.,-]+")
        self.formulas = re.compile(r"\$.+\$", re.IGNORECASE)
        """Stop words list, English + punctuation"""
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update(['.', ',', '"', "'", '?', '!', ':',
                                '$', '%', ';', '(', ')', '[', ']',
                                '{', '}', '``', '--', '""', "'s",
                                '\\', 'one', 'two', 'three', 'four',
                                'five', 'six', 'seven', 'eight',
                                'nine', 'ten', 'new', 'novel',
                                'we', 'us', 'also'])
        """
        Stemming
        URL: http://www.nltk.org/api/nltk.stem.html
        """
        self.porter = PorterStemmer()
        """POS tags to be removed"""
        self.tags_to_remove = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
                               'CC', 'CD', 'DT', 'EX', 'IN', 'JJR', 'JJS',
                               'LS', 'PDT', 'SYM', 'WDT', 'WP', 'WP$', 'WRB',
                               'TO', 'UH', 'POS', 'PRP', 'PRP$']
        # NOTE: comparative and superlative adjectives removal
        """Lemmatization"""
        self.lemmatizer = WordNetLemmatizer()

        self.n_samples = 10
        self.n_features = 1000
        self.n_top_words = 10

        # self.tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
        #                                      max_features=self.n_features,
        #                                      stop_words='english')

    def __deepcopy__(self):
        return FullTextTransformer()

    def __foo(self, data):
        abstract = data.replace("\n", " ").strip()

        """Math formulas removal"""
        abstract = self.formulas.sub("", abstract)
        abstract = self.numbers.sub("", abstract)
        # print abstract
        # print

        """Word tokenizing"""
        tokens = word_tokenize(abstract)
        # print tokens
        # print

        """Punctuation and stopwords removal"""
        filtered_words = [word.lower()
                          for word in tokens
                          if word.lower() not in self.stop_words]
        # print filtered_words
        # print

        """
        Tagging
        POS tags
        URL: https://www.ling.upenn.edu
        /courses/Fall_2003/ling001/penn_treebank_pos.html
        """
        tagged = pos_tag(filtered_words)
        # print tagged
        # print

        """Tags filtering"""
        filtered_tags = [(word, tag)
                         for word, tag in tagged
                         if tag not in self.tags_to_remove]
        # print filtered_tags
        # print

        """Stemming"""
        # stemmed = [porter.stem(word) for word, tag in filtered_tags]
        stemmed = [word for word, tag in filtered_tags]
        # print stemmed
        # print

        """Lemmatization"""
        lemmatized = [self.lemmatizer.lemmatize(word) for word in stemmed]
        return " ".join(lemmatized)

    def transform(self, X, *_):
        return np.vectorize(self.__foo)(X)

    def fit(self, *_):
        return self


class FullTextPipeline(BasePipeline):
    def __init__(self, estimator):
        self.pipe = Pipeline([('fulltext transformer', FullTextTransformer()),
                              ('hashing vectorizer', HashingVectorizer()),
                              ('estimator', estimator)])


class WordEmbeddingPipeline(BasePipeline):
    def __init__(self):
        # path = get_resource_path()
        # model_path = os.path.join(path, "data/en_1000_no_stem/en.model")
        # model = Word2Vec.load(model_path)
        # print(model.similarity('woman', 'man'))
        pass


def main():
    """
    """
    resource_path = get_resource_path()
    folder = os.path.join(resource_path, 'data/amazon')
    datafile_train = 'Electronics_5.json'
    datafile_test = 'Books_5.json'
    X_train, y_train = [], []
    X_test, y_test = [], []

    for line in open(os.path.join(folder, datafile_train)):
        content = json.loads(line)
        X_train.append(content["reviewText"])
        y_train.append(float(content["overall"]))

    for line in open(os.path.join(folder, datafile_test)):
        content = json.loads(line)
        X_test.append(content["reviewText"])
        y_test.append(float(content["overall"]))

    size = 100

    # pipeline = FullTextPipeline(RandomForest())
    pipeline = HashingPipeline(RandomForest())
    # pipeline = TfIdfPipeline(RandomForest())
    model = pipeline.fit(X_train[:size], y_train[:size])

    shift_detector = SklearnDataShiftDetector(model, n_bins=1000)
    shift_detector.iteration(X_train[:100])
    shift_detector.iteration(X_test[:100])
    print(shift_detector.data_is_shifted())


if __name__ == "__main__":
    main()
