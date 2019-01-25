#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

""""""

import numpy as np

from profilers import DataFrameProfiler
from analyzers import DataType
from messages import Message as mes


class Selector:
    def __init__(self):
        pass


class RandomSelector(Selector):
    class __RandomSelector:
        def __init__(self, column_fraction=.2, row_fraction=.1):
            # Selector.__init__(self)
            self.column_fraction = column_fraction
            self.row_fraction = row_fraction

        def on(self, data, columns=None):
            return self.run(data, columns)

        def run(self, data, columns=None):
            rows, cols = data.shape
            list_of_cols = list(data.columns)
            if columns is None:
                # random 'corrupting'
                columns = np.random.choice(range(cols),
                                           int(np.ceil(
                                               self.column_fraction*cols)),
                                           replace=False)
            elif isinstance(columns, str):
                profiles = DataFrameProfiler().on(data).profiles
                if columns == 'string':
                    columns = [list_of_cols.index(p.column_name)
                               for p in profiles if p.dtype == DataType.STRING]
                elif columns == 'numeric':
                    columns = [list_of_cols.index(p.column_name)
                               for p in profiles if p.dtype
                               in [DataType.INTEGER, DataType.FLOAT]]
                else:
                    columns = np.random.choice(range(cols),
                                               int(np.ceil(
                                                   self.column_fraction*cols)),
                                               replace=False)
            else:
                columns = [list_of_cols.index(col) for col in columns]
            tmp = dict({})
            for col in columns:
                tmp[col] = np.random.choice(
                    range(rows),
                    int(np.ceil(self.row_fraction*rows)),
                    replace=False)
            return tmp

    instance = None

    def __init__(self, column_fraction=.2, row_fraction=.1):
        if not RandomSelector.instance:
            RandomSelector.instance = (RandomSelector
                                       .__RandomSelector(column_fraction,
                                                         row_fraction))
        else:
            RandomSelector.instance.column_fraction = column_fraction
            RandomSelector.instance.row_fraction = row_fraction

    def __getattr__(self, name):
        return getattr(self.instance, name)


class PairSelector(Selector):
    class __PairSelector:
        def __init__(self, row_fraction=.1):
            # Selector.__init__(self)
            self.row_fraction = row_fraction

        def on(self, data, columns=None):
            return self.run(data, columns)

        def run(self, data, columns=None):
            rows, cols = data.shape
            list_of_cols = list(data.columns)
            if columns is None:
                # random 'corrupting'
                # columns = np.random.choice(range(cols), 2, replace=False)
                columns = 'string'
            if isinstance(columns, str):
                modes = ['string', 'numeric']
                assert columns in modes, mes().wrong_value % columns
                profiles = DataFrameProfiler().on(data).profiles
                if columns == 'string':
                    tmp = [DataType.STRING]
                elif columns == 'numeric':
                    tmp = [DataType.INTEGER, DataType.FLOAT]
                columns = [list_of_cols.index(p.column_name)
                           for p in profiles if p.dtype in tmp]
            else:
                assert isinstance(columns, list), mes().wrong_value % columns
                columns = [list_of_cols.index(col) for col in columns]
            columns = np.random.choice(columns, 2, replace=False)
            tmp = dict({})
            rows = np.random.choice(range(rows),
                                    int(np.ceil(self.row_fraction*rows)),
                                    replace=False)
            assert np.unique(columns).shape[0] > 1, "! not enough columns"
            for col in columns:
                tmp[col] = rows
            return tmp

    instance = None

    def __init__(self, column_fraction=.2, row_fraction=.1):
        if not PairSelector.instance:
            PairSelector.instance = (PairSelector
                                     .__PairSelector(row_fraction))
        else:
            PairSelector.instance.row_fraction = row_fraction

    def __getattr__(self, name):
        return getattr(self.instance, name)


def main():
    """
    """
    pass


if __name__ == "__main__":
    main()
