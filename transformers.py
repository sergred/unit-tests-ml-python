#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

""""""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np


class OrdinalScaleTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, array):
        self.array = array

    def transform(self, X, *_):
        return np.vectorize(lambda x: self.array.index(x))(X).reshape(-1, 1)

    def fit(self, *_):
        return self


class OneHotEncodingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, val='auto'):
        self.val = val
        self.n_values = len(self.val)
        self.encoder = OneHotEncoder(categories=[self.val],
                                     n_values=[self.n_values],
                                     sparse=False)

    def transform(self, X, *_):
        return self.encoder.fit_transform(X.values.reshape(-1, 1))

    def fit(self, *_):
        return self


class LambdaEncodingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, func):
        self.func = func

    def transform(self, X, *_):
        return np.vectorize(self.func)(X).reshape(-1, 1)

    def fit(self, *_):
        return self


class Imputer(BaseEstimator, TransformerMixin):
    def __init__(self, values):
        self.values = values
        self.imputer = SimpleImputer(missing_values=self.values,
                                     strategy='most_frequent')

    def transform(self, X, *_):
        return self.imputer.fit_transform(X.values.reshape(-1, 1))

    def fit(self, *_):
        return self


def main():
    """
    """
    pass


if __name__ == "__main__":
    main()
