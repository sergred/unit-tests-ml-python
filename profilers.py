#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

""""""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as p
from collections import defaultdict
from enum import Enum
import pandas as pd
import numpy as np
import inspect
import pickle

from analyzers import DataFrameAnalyzer

class ErrorType(Enum):
    UNDEFINED = 0
    MISSING_VALUE = 1
    ANOMALY = 2
    TYPO = 3
    INCOMPLETE = 4
    DUPLICATE = 5
    INTEGRITY = 6


class Warning:
    def __init__(self, error_type, message):
        self.error_type = error_type
        self.message = message

    def __repr__(self):
        return self.message


class ColumnProfile:
    def __init__(self, column_name,
                 dtype, num_missing,
                 num_unique, histogram):
        self.column_name = column_name
        self.dtype = dtype
        self.num_missing = num_missing
        self.num_unique = num_unique
        self.histogram = histogram

    def str(self):
        return self.__repr__()

    def __repr__(self):
        return """
        column    : %s
        type      : %s
        missing   : %d
        unique    : %d
        histogram : %s
        """ % (self.column_name, self.dtype, self.num_missing,
               self.num_unique, self.histogram)


class Profiler:
    def __init__(self):
        pass


class DataFrameProfiler(Profiler):
    def __init__(self):
        self.profiles = []

    def __enter__(self):
        return self

    def __exit__(self):
        return False

    def on(self, data):
        return self.run(data, columns=None)

    def run(self, data, columns=None):
        self.analyzer = DataFrameAnalyzer()
        self.analyzer.run(data, columns)
        for i, col in enumerate(data.columns):
            stats = self.analyzer.stats.iloc[:, i].values
            profile = ColumnProfile(col, self.analyzer.dtypes[i],
                                   np.sum(pd.isnull(data.iloc[i])),
                                   int(stats[1]), self.analyzer.histograms[i])
            self.profiles.append(profile)
        return self.profiles


class PipelineProfiler(Profiler):
    def __init__(self):
        pass


class TensorflowPipelineProfiler(Profiler):
    def __init__(self):
        pass


class SklearnPipelineProfiler(Profiler):
    def __init__(self):
        self.profiles = defaultdict(list)
        self.rules = dict({
            'OneHotEncodingTransformer': lambda x: vars(x.encoder),
            'OrdinalScaleTransformer': lambda x: vars(x),
            'ColumnTransformer': lambda x: self.__analyzeCT(x),
            'StandardScaler': lambda x: [x.mean_, x.var_],
            'SimpleImputer': lambda x: x.statistics_,
            'RandomForest': lambda x: vars(pickle.loads(x.dump))
        })

    def __analyzeCT(self, func):
        for each in func.transformers:
            new_name, transformer, name = each
            print("%s\n%s\n%s\n" % (name, transformer.__class__.__name__,
                                    self.derive_info(transformer)))

    def derive_info(self, transformer):
        name = transformer.__class__.__name__
        return self.rules[name](transformer) if name in self.rules else None

    def on(self, pipeline):
        return self.run(pipeline)

    def run(self, pipeline):
        assert isinstance(pipeline, p), "sklearn.pipeline.Pipeline required"
        # print(pipeline)
        for step in pipeline.steps:
            name, func = step
            print("%s\n%s\n" % (name, self.derive_info(func)))


def main():
    """
    """
    from pipelines import CreditGPipeline
    from models import RandomForest
    dataframe = pd.read_csv('resources/data/dataset_31_credit-g.csv')
    DataFrameProfiler().on(dataframe)
    pipeline = CreditGPipeline()
    SklearnPipelineProfiler().on(pipeline.with_estimator(RandomForest(40)))


if __name__ == "__main__":
    main()
