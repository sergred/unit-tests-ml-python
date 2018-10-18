#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

""""""

from copy import deepcopy
import numpy as np

class ErrorGenerator:
    def __init__(self):
        pass

    def on(self, data):
        return self.run(data)

    def run(self, data, columns=None):
        return data


class Anomalies(ErrorGenerator):
    def __init__(self, mode):
        ErrorGenerator.__init__(self)
        self.mode = mode
        pass

    def get_anomaly(self, column_stats, column_dtype):
        anomaly = 0
        if column_dtype == 'object':
            anomaly = "anomaly"
            if column_stats['unique'] > 0:
                pass
        elif column_dtype == 'int64':
            pass
        elif column_dtype == 'float64':
            pass
        else:
            pass
        return anomaly

    def run(self, data, columns=None):
        tmp = deepcopy(data)
        rows, cols = tmp.shape
        stats = data.describe(include='all')
        if not columns:
            # random 'corrupting'
            fraction = .4
            columns = np.random.choice(range(cols), int(np.ceil(fraction*cols)), replace=False)
        for col in columns:
            column_stats = stats.iloc[:, col]
            print(column_stats)
            tmp.iloc[np.random.choice(range(rows), int(np.ceil(fraction*rows)), replace=False), col] = self.get_anomaly(column_stats, data.dtypes[col])
        return tmp


class MissingValues(ErrorGenerator):
    def __init__(self):
        ErrorGenerator.__init__(self)
        pass

    def run(self, data, columns=None):
        tmp = deepcopy(data)
        rows, cols = tmp.shape
        if not columns:
            # random 'corrupting'
            fraction = .4
            columns = np.random.choice(range(cols), int(np.ceil(fraction*cols)), replace=False)
        for col in columns:
            tmp.iloc[np.random.choice(range(rows), int(np.ceil(fraction*rows)), replace=False), col] = None
        return tmp


def main():
    """
    """
    pass


if __name__ == "__main__":
    main()
