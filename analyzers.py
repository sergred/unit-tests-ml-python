#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

""""""

from collections import Counter
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
        self.discrete_threshold = 120
        self.all_stat_items = ['count']
        self.obj_stat_items = ['unique', 'top', 'freq']
        self.date_stat_items = ['first', 'last']
        self.num_stat_items = ['mean', 'std',
                               'min', '25%', '50%', '75%', 'max']
        self.stat_items_strategy = dict({
            1: self.all_stat_items,
            3: self.all_stat_items + self.date_stat_items,
            4: self.all_stat_items + self.obj_stat_items,
            6: self.all_stat_items + self.obj_stat_items
            + self.date_stat_items,
            8: self.all_stat_items + self.num_stat_items,
            10: self.all_stat_items + self.date_stat_items
            + self.num_stat_items,
            11: self.all_stat_items + self.obj_stat_items
            + self.num_stat_items,
            13: self.all_stat_items + self.obj_stat_items
            + self.date_stat_items + self.num_stat_items
        })
        self.compute = dict({
            'count': lambda x: np.array([x.shape[0]]*x.shape[1]),
            'unique': lambda x: np.unique(x, axis=0),
            'top': lambda x: [None]*x.shape[1],
            'freq': lambda x: [None]*x.shape[1],
            'first': lambda x: [None]*x.shape[1],
            'last': lambda x: [None]*x.shape[1],
            'mean': lambda x: np.mean(x, axis=0),
            'std': lambda x: np.std(x, axis=0),
            'min': lambda x: np.min(x, axis=0),
            '25%': lambda x: [None]*x.shape[1],
            '50%': lambda x: [None]*x.shape[1],
            '75%': lambda x: [None]*x.shape[1],
            'max': lambda x: np.max(x, axis=0)
        })

    def on(self, dataframe):
        return self.run(dataframe)

    def _get_histogram(self, dataframe, column_idx=None):
        if column_idx is None:
            column_idx = range(dataframe.shape[1])
        column_idx = np.array(column_idx)
        stats = self._get_stats(column_idx, ['unique', 'min', 'max'])
        histograms = []
        for idx in column_idx:
            tmp = None
            if stats[0, idx] <= self.discrete_threshold:
                tmp = list(Counter(dataframe.iloc[:, idx].values).items())
            else:
                if self.dtypes[idx] == DataType.STRING:
                    # TODO: word embeddings?
                    tmp = None
                else:
                    bins = np.linspace(stats[1, idx], stats[2, idx],
                                       self.discrete_threshold)
                    # bins = np.arange(stats[1, idx], stats[2, idx], step)
                    digitized = np.digitize(
                        dataframe.iloc[:, idx].astype(np.float64).values, bins)
                    bin_height = [np.sum(digitized == i)
                                  for i in range(1, len(bins))]
                    tmp = list(zip(bins[:-1], bins[1:], bin_height))
                    tmp = [item for item in tmp if item[2] != 0]
            histograms.append(tmp)
        return np.array(histograms)

    def _get_stats(self, columns, keys):
        # assert set(keys) < set(self.stat_items.keys()), "Wrong key"
        tmp = []
        for key in keys:
            if key not in self.stat_items.keys():
                tmp.append(self.compute[key](columns))
            else:
                tmp.append(self.stats.iloc[self.stat_items[key],
                                           columns].values)
        # print(self.stats.shape)
        # print(self.stats)
        # print([self.stat_items[k] for k in list(keys)])
        # print(columns)
        return np.array(tmp)

    def _get_column_scale(self, columns):
        if pd.isnull(self._get_stats(columns, ['unique', 'top', 'freq'])).all():
            # if (column_stats.min == 0.) and (columns.stats.max == 1.):
            #     scale = DataScale.RATIO
            scale = DataScale.ORDINAL
        elif not pd.isnull(self._get_stats(columns, ['unique', 'top', 'freq'])).all():
            scale = DataScale.NOMINAL
        else:
            scale = DataScale.UNDEFINED
        return scale

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
        self.stats = dataframe.describe(include='all')
        stat_items = self.stat_items_strategy[self.stats.shape[0]]
        self.stat_items = dict(zip(stat_items, range(len(stat_items))))
        self.columns = dataframe.columns if not columns else columns
        self.dtypes = np.array([self._get_column_type(col)
                                for col in dataframe.dtypes])
        for idx, type in enumerate(self.dtypes):
            if type is DataType.INTEGER:
                self.stats.iloc[self.stat_items['unique'], idx] = np.unique(
                    dataframe.iloc[:, idx]).shape[0]
        # for col in self.columns: column_stats = self.stats.loc[:, col]
        self.histograms = self._get_histogram(dataframe)
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
