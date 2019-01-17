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
        # print('ordinal')
        # return X.apply(lambda x: self.array.index(x))
        return (np.vectorize(lambda x: self.array.index(x))
                (X.values.reshape(-1, 1)))

    def fit(self, *_):
        return self


class OneHotEncodingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, val):
        self.val = val
        self.n_values = len(self.val)
        self.encoder = OneHotEncoder(categories=[self.val],
                                     n_values=[self.n_values],
                                     sparse=False)

    def transform(self, X, *_):
        # print('onehot')
        return self.encoder.fit_transform(X.values.reshape(-1, 1))

    def fit(self, *_):
        return self


class LambdaEncodingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, func):
        self.func = func

    def transform(self, X, *_):
        # print('lambda')
        # return X.apply(self.func)
        return (np.vectorize(lambda x: self.func(x))
                (X.values.reshape(-1, 1)))

    def fit(self, *_):
        return self


class Imputer(BaseEstimator, TransformerMixin):
    def __init__(self, values=np.NaN, strategy='mean'):
        self.values = values
        self.strategy = strategy
        self.imputer = SimpleImputer(missing_values=self.values,
                                     strategy=self.strategy)

    def transform(self, X, *_):
        print('custom imputer')
        return self.imputer.fit_transform(X.values.reshape(-1, 1))

    def fit(self, *_):
        return self


class DenseTransformer(TransformerMixin):
    def fit(self, *_):
        return self

    def transform(self, X, *_):
        return X.todense() if 'todense' in dir(X) else X


class SklearnAutomatedFilter(TransformerMixin):
    def fit(self, *_):
        return self

    def transform(self, X, *_):
        pass


def main():
    """
    """
    pass


if __name__ == "__main__":
    main()
