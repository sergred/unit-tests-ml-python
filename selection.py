#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

""""""

from analyzers import DataFrameAnalyzer, DataType
from copy import deepcopy
import numpy as np

class Selector:
    def __init__(self):
        pass


class RandomSelector(Selector):
    class __RandomSelector:
        def __init__(self, column_fraction=.2, row_fraction=.1):
            Selector.__init__(self)
            self.column_fraction = column_fraction
            self.row_fraction = row_fraction

        def on(self, data, columns=None, column_type=None):
            return self.run(data, columns, column_type)

        def run(self, data, columns=None, column_type=None):
            rows, cols = data.shape
            if columns is None:
                if column_type is None:
                    # random 'corrupting'
                    columns = np.random.choice(range(cols),
                                               int(np.ceil(self.column_fraction*cols)), replace=False)
                else:
                    columns = DataFrameAnalyzer(data).get_column_idx(column_type)
            else:
                columns = [list(data.columns).index(col) for col in columns]
            row_ids = np.random.choice(range(rows), int(np.ceil(self.row_fraction*rows)), replace=False)
            return dict([(col, row_ids) for col in columns])

    instance = None

    def __init__(self, column_fraction=.2, row_fraction=.1):
        if not RandomSelector.instance:
            RandomSelector.instance = RandomSelector.__RandomSelector(column_fraction, row_fraction)
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
