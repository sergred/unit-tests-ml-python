#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

""""""

from enum import Enum
import pandas as pd
import numpy as np

class DataScale(Enum):
    UNDEFINED = 0
    NOMINAL = 1 # categorical
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
    def __init__(self, data):
        Analyzer.__init__(self)
        self.data = data
        self.stats = None
        stat_items = ['count', 'unique', 'top', 'freq', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
        self.stat_items = dict(zip(stat_items, range(len(stat_items))))
        self.run(self.data)
        pass

    def on(self, dataframe):
        return self.run(dataframe)

    def _get_stats(self, column, keys):
        assert set(keys) < set(self.stat_items.keys()), "Wrong key"
        return self.stats.iloc[[self.stat_items[k] for k in list(keys)], column].values

    def _get_column_scale(self, column_name):
        if pd.isnull(self._get_stats(column_name, ['unique', 'top', 'freq'])).all():
            # if (column_stats.min == 0.) and (columns.stats.max == 1.):
            #     scale = DataScale.RATIO
            scale = DataScale.ORDINAL
        elif not pd.isnull(self._get_stats(column_name, ['unique', 'top', 'freq'])).all():
            scale = DataScale.NOMINAL
        else:
            scale = DataScale.UNDEFINED
        return scale

    def _get_column_type(self, column_type, column=None, column_stats=None):
        if column_type == 'object':
            dtype = DataType.STRING
        elif np.issubdtype(column_type, np.signedinteger):
            dtype = DataType.INTEGER
        elif np.issubdtype(column_type, np.floating):
            if not column and not column_stats:
                dtype = DataType.FLOAT
            elif column is not None and column_stats is not None:
                if column_stats.min.is_integer() and column_stats.min.is_integer():
                    if map(lambda x: x.is_integer(), column).all():
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
        self.columns = dataframe.columns if not columns else columns
        self.dtypes = np.array([self._get_column_type(col) for col in dataframe.dtypes])
        for col in self.columns:
            column_stats = self.stats.loc[:, col]
            # print(col)
            # print(self._get_column_scale(col))
            # print(self._get_column_type(self.dtypes[col]))
            # print(column_stats)
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
