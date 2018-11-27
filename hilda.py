#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

""""""

from sklearn.model_selection import train_test_split as split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

from pipelines import CreditGPipeline
from error_generation import ExplicitMissingValues
from models import RandomForest

class Check:
    def __init__(self, assumption, message):
        self.assumption = assumption
        self.message = message


class Test:
    def __init__(self):
        self.checks = []

    def __iter__(self):
        return iter(self.checks)

    def _add(self, assumption, message):
        self.checks.append(Check(assumption, message))
        return self

    def has_size(self, size):
        return self._add(lambda x: x.shape[0] == size,
                         "DataFrame does not have %d rows" % size)

    def is_complete(self, column):
        return self._add(lambda x: x[column].notna().all(),
                         "Column %s is not complete" % column)

    def is_unique(self, column):
        return self._add(lambda x: np.unique(x[column]).shape[0] == x.shape[0],
                         "Values in column %s not not unique" % column)


class TestSuite:
    def __init__(self, data_profile=None, pipeline_profile=None):
        self.data = None
        self.tests = []
        self.data_profile = data_profile
        self.pipeline_profile = pipeline_profile

    def on(self, data):
        self.data = data
        return self

    def add(self, test):
        self.tests.append(test)
        return self

    def run(self):
        assert self.data is not None, "Call TestSuite on(data) method first."
        messages = []
        for test in self.tests:
            for check in test:
                if not check.assumption(self.data):
                    messages.append(check.message)
        return messages


def main():
    """
    """
    data = pd.read_csv('resources/data/dataset_31_credit-g.csv')
    print(data.shape)
    print(data.columns)

    target = "class"
    X, y = data[[col for col in data.columns if col != target]], data[target]
    X_train, X_test, y_train, y_test = split(X, y,
                                             test_size=0.2,
                                             random_state=0)

    pipeline = CreditGPipeline()
    classifier = RandomForest(size=40)
    model = pipeline.with_estimator(classifier).fit(X_train, y_train)

    prediction = model.predict(X_test)
    print(accuracy_score(y_test, prediction))

    # data_profile = profile(X_train)
    # pipeline_profile = inspect(model)
    data_profile = None
    pipeline_profile = None
    suite = TestSuite(pipeline_profile, data_profile)

    suite.add(Test()
              .is_complete('checking_status'))

    warnings = suite.on(X_test).run()

    if warnings:
        print("======= WARNINGS =======")
    for warn in warnings:
        print(warn)

    error_generator = ExplicitMissingValues()
    corrupted_X_test = error_generator.run(X_test, ['checking_status'])

    warnings = suite.on(corrupted_X_test).run()

    if warnings:
        print("======= WARNINGS =======")
    for warn in warnings:
        print(warn)



if __name__ == "__main__":
    main()
