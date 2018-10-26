#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.metrics import accuracy_score
from tabulate import tabulate
import pandas as pd
import numpy as np
import os

from models import DecisionTree, RandomForest, ExtremelyRandomizedTrees, XGB
from pipelines import AbalonePipeline, CreditGPipeline, WineQualityPipeline, WineQualityMissingPipeline
from error_generation import MissingValues, Anomalies, Typos
from settings import get_resource_path

np.random.seed(0)

class Table:
    def __init__(self, rows, columns, subrows, subcolumns):
        self.row_header_len, self.col_header_len = 2, 2
        self.rows_len, self.cols_len = len(rows), len(columns)
        self.subrows_len, self.subcols_len = len(subrows) + 1, len(subcolumns)
        self.dataset_baseline_title = 'baseline'
        self.table = np.zeros((self.rows_len*self.subrows_len+self.row_header_len, self.cols_len*self.subcols_len+self.col_header_len), dtype=object)
        self.table[:self.col_header_len, :self.row_header_len] = np.array([['', 'classifiers'], ['datasets', 'tests\\anomalies']])
        self.table[0, self.row_header_len:] = np.array([list(columns)[i//self.subcols_len] for i in range(self.cols_len*self.subcols_len)])
        self.table[1, self.row_header_len:] = np.array(list(subcolumns)*self.cols_len)
        self.table[self.col_header_len:, 0] = np.array([list(rows)[i//self.subrows_len] if i%self.subrows_len == 0 else '' for i in range(self.rows_len*self.subrows_len)])
        self.table[self.col_header_len:, 1] = np.array(([self.dataset_baseline_title] + list(subrows))*self.rows_len)

    def update(self, data, i, j, k=-1, l=-1):
        if k == -1 and l == -1:
            z = self.row_header_len + self.subcols_len*j
            self.table[self.col_header_len + self.subrows_len*i, z:z+self.subcols_len] = data
        elif k != -1 and l != -1:
            self.table[self.col_header_len + self.subrows_len*i + k + 1, self.row_header_len + self.subcols_len*j + l] = data
        else:
            raise Exception('wrong subs')

    def save(self, filename):
        np.savetxt(filename, self.table, delimiter=",", fmt="%s")

    def show(self):
        res = tabulate(self.table, tablefmt='psql')
        print(res)
        return res


def main():
    resource_folder = get_resource_path()
    # for dataset_name in sorted(os.listdir(folder)):
    #     if dataset_name.endswith('.csv'):
    #         print(dataset_name[:-4])

    pipelines = {'credit-g': ('dataset_31_credit-g.csv', 'class', CreditGPipeline()),
                 'wine-quality': ('wine-quality-red.csv', 'class', WineQualityPipeline()),
                 'wq-missing': ('wine-quality-red.csv', 'class', WineQualityMissingPipeline())}
    #            'abalone': ('abalone.csv', 'Rings', AbalonePipeline),

    classifiers = {'dtc': DecisionTree(),
                   'rfc': RandomForest(size=40),
                   'ertc': ExtremelyRandomizedTrees(size=40),
                   'xgb': XGB()}

    error_gens = {'numeric anomalies': Anomalies(),
                  'typos': Typos(),
                  'missing values': MissingValues()}

    # 'implicit missing values': ImplicitMissingValues(),
    # 'explicit missing values': ExplicitMissingValues()}

    tests = {'core': "mock"}

    results = Table(rows=pipelines.keys(), columns=classifiers.keys(), subrows=tests.keys(), subcolumns=error_gens.keys())

    for pipe_idx, pipe in enumerate(sorted(pipelines.items())):
        name, content = pipe
        filename, target, pipeline = content
        data = pd.read_csv(os.path.join(resource_folder, "data", filename))
        print(name)
        # print(data.shape)
        # print(data.info(verbose=True))

        X, y = data[[col for col in data.columns if col != target]], data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # ss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)

        for classifier_idx, content in enumerate(sorted(classifiers.items())):
            cls_name, classifier = content
            # print(cls_name)

            # for train_index, test_index in ss.split(X, y):
            #     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            #     y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model = pipeline.with_estimator(classifier).fit(X_train, y_train)
            prediction = model.predict(X_test)
            base_score = accuracy_score(y_test, prediction)
            results.update(base_score, pipe_idx, classifier_idx)

            for err_gen_idx, err_gen in enumerate(error_gens.values()):
                # corrupted_X_train = err_gen.on(X_train)
                # model = pipeline.with_estimator(classifier).fit(corrupted_X_train, y_train)
                # prediction = model.predict(X_test)
                # res = "%.4f" % round(accuracy_score(y_test, prediction) - base_score, 4)
                try:
                    corrupted_X_train = err_gen.on(X_train)
                    model = pipeline.with_estimator(classifier).fit(corrupted_X_train, y_train)
                    prediction = model.predict(X_test)
                    res = "%.4f" % round(accuracy_score(y_test, prediction) - base_score, 4)
                except Exception as e:
                    print("%s: %s" % (err_gen.__class__, e))
                    res = 'Fail'

                results.update(res, pipe_idx, classifier_idx, 0, err_gen_idx)

    results.show()
    results.save(os.path.join(resource_folder, "results/matrix.csv"))


if __name__ == "__main__":
    main()
