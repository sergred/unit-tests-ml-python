#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

""""""

from collections import Counter, defaultdict
from enum import Enum
import pandas as pd
import numpy as np


class DataScale(Enum):
    UNDEFINED = 0
    NOMINAL = 1  # categorical
    ORDINAL = 2
    INTERVAL = 3
    RATIO = 4


class DataType(Enum):
    UNDEFINED = 0
    NONE = 1
    STRING = 2
    INTEGER = 3
    FLOAT = 4
    OBJECT = 5


class Column:
    def __init__(self):
        pass


class NominalColumn(Column):
    def __init__(self):
        pass


class OrdinalColumn(Column):
    def __init__(self):
        pass


class Analyzer:
    def __init__(self):
        pass


class DataFrameAnalyzer(Analyzer):
    def __init__(self):
        Analyzer.__init__(self)
        self.stats = None
        self.discrete_threshold = 60
        self.compute = dict({
            'count': lambda x: self._compute_default(x, x.shape[0]),
            'unique': lambda x: self._compute_unique(x),
            'top': lambda x: self._compute_default(x, None),
            'freq': lambda x: self._compute_default(x, None),
            'first': lambda x: self._compute_default(x, None),
            'last': lambda x: self._compute_default(x, None),
            'mean': lambda x: np.mean(x, axis=0),
            'std': lambda x: np.std(x, axis=0),
            'min': lambda x: np.min(x, axis=0),
            '25%': lambda x: self._compute_default(x, None),
            '50%': lambda x: self._compute_default(x, None),
            '75%': lambda x: self._compute_default(x, None),
            'max': lambda x: np.max(x, axis=0)
        })

    def _compute_default(self, x, value):
        return [value]*(x.shape[1] if len(x.shape) > 1 else 1)

    def _compute_unique(self, x):
        return [len(Counter(x.iloc[:, i]).items()) for i in range(x.shape[1])]

    def on(self, dataframe):
        return self.run(dataframe)

    def _get_histogram(self, columns=None):
        if columns is None:
            columns = self.columns
        # stats = self._get_stats(column_idx, ['unique', 'min', 'max'])
        histograms = []
        for idx, col in enumerate(columns):
            tmp = None
            if self.stats.loc['unique', col] <= self.discrete_threshold:
                # tmp = list(Counter(self.data[col].values).items())
                # tmp = np.unique(self.data[col].values)
                column = sorted(self.data[col].values)
                tmp = []
                current = column[0]
                count = 1
                for i in range(1, len(column)):
                    if column[i] == current:
                        count += 1
                    else:
                        tmp.append((current, count))
                        count = 1
                        current = column[i]
                tmp.append((current, count))
            else:
                if self.dtypes[idx] == DataType.STRING:
                    # TODO: word embeddings?
                    tmp = None
                else:
                    bins = np.linspace(self.stats.loc['min', col],
                                       self.stats.loc['max', col],
                                       self.discrete_threshold)
                    digitized = np.digitize(
                        self.data.iloc[:, idx].astype(np.float64).values, bins)
                    bin_height = [np.sum(digitized == i)
                                  for i in range(1, len(bins))]
                    tmp = list(zip(bins[:-1], bins[1:], bin_height))
                    tmp = [item for item in tmp if item[2] != 0]
            histograms.append(tmp)
        return np.array(histograms)

    def _get_stats(self, columns):
        column_idx = [self.columns.index(col) for col in columns]
        tmp = []
        for key in self.compute:
            if key in self.stat_items:
                to_append = list(
                    self.stats.iloc[self.stat_items[key], column_idx].values)
            else:
                to_append = self.compute[key](self.data.iloc[:, column_idx])
            tmp.append(to_append)
        return pd.DataFrame(tmp, columns=self.columns,
                            index=self.compute.keys())

    def _get_column_scale(self, columns):
        scales = []
        for col in columns:
            condition = self.stats.loc['unique', col] > self.discrete_threshold
            scale = DataScale.ORDINAL if condition else DataScale.NOMINAL
            scales.append(scale)
        return scales

    def _get_column_type(self, column_type, columns=None, column_stats=None):
        if column_type == 'object':
            dtype = DataType.STRING
        elif np.issubdtype(column_type, np.signedinteger):
            dtype = DataType.INTEGER
        elif np.issubdtype(column_type, np.floating):
            if not columns and not column_stats:
                dtype = DataType.FLOAT
            elif columns is not None and column_stats is not None:
                if (column_stats.min.is_integer()
                        and column_stats.max.is_integer()):
                    if map(lambda x: x.is_integer(), columns).all():
                        dtype = DataType.INTEGER
                    else:
                        dtype = DataType.FLOAT
                else:
                    dtype = DataType.FLOAT
        else:
            dtype = DataType.UNDEFINED
        return dtype

    def get_column_idx(self, type):
        return np.arange(self.columns.shape[0])[self.dtypes == type]

    def run(self, dataframe, columns=None):
        self.data = dataframe
        self.columns = list(self.data.columns) if not columns else columns
        self.stats = self.data.describe(include='all')
        self.stat_items = dict(map(lambda t: (t[1], t[0]),
                                   enumerate(self.stats.index.values)))
        self.stats = self._get_stats(self.columns)
        self.scales = self._get_column_scale(self.columns)
        self.dtypes = np.array([self._get_column_type(col)
                                for col in self.data.dtypes])
        self.histograms = self._get_histogram()
        return self


def main():
    """
    """
    from settings import get_resource_path
    import os
    resource_folder = get_resource_path()
    filename = 'dataset_31_credit-g.csv'
    data = pd.read_csv(os.path.join(resource_folder, "data", filename))
    analyzer = DataFrameAnalyzer()
    analyzer.on(data)


if __name__ == "__main__":
    main()
