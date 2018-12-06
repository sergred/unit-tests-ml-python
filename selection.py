#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

""""""

import numpy as np

from profilers import DataFrameProfiler
from analyzers import DataType


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
                # columns = (DataFrameAnalyzer()
                #            .on(data)
                #            .get_column_idx(column_type))
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
    def __init__(self):
        pass


def main():
    """
    """
    pass


if __name__ == "__main__":
    main()
