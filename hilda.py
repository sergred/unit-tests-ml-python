#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

""""""

from sklearn.model_selection import train_test_split as split
from sklearn.metrics import accuracy_score
import pandas as pd

from pipelines import WineQualityMissingPipeline
# from pipelines import CreditGPipeline
from profilers import SklearnPipelineProfiler
from test_suite import AutomatedTestSuite, TestSuite, Test
from error_generation import ExplicitMissingValues
from models import RandomForest


def main():
    """
    """
    # data = pd.read_csv('resources/data/dataset_31_credit-g.csv')
    data = pd.read_csv('resources/data/wine-quality/wine-quality-red.csv')
    print(data.shape)
    print(data.columns)

    target = "class"
    X, y = data[[col for col in data.columns if col != target]], data[target]
    X_train, X_test, y_train, y_test = split(X, y,
                                             test_size=0.2,
                                             random_state=0)

    # pipeline = CreditGPipeline()
    pipeline = WineQualityMissingPipeline()
    classifier = RandomForest(size=40)
    model = pipeline.with_estimator(classifier).fit(X_train, y_train)

    prediction = model.predict(X_test)
    print(accuracy_score(y_test, prediction))

    suite = TestSuite()
    automated_suite = AutomatedTestSuite()
    pipeline_profile = SklearnPipelineProfiler().on(model)

    suite.add(Test()
              .is_complete('volatile_acidity'))

    warnings = suite.on(X_test)

    print("*** TEST_SUITE, X_TEST")
    if warnings and (len(warnings) != 0):
        print("======= WARNINGS =======")
        for warn in warnings:
            print(warn)

    error_generator = ExplicitMissingValues()
    corrupted_X_test = error_generator.run(X_test, ['volatile_acidity'])

    warnings = suite.on(corrupted_X_test)

    print("*** TEST_SUITE, CORRUPTED_X_TEST")
    if warnings and (len(warnings) != 0):
        print("======= WARNINGS =======")
        for warn in warnings:
            print(warn)

    print()

    tests, warnings = automated_suite.run(corrupted_X_test, pipeline_profile)

    print("*** AUTOMATED_TEST_SUITE, CORRUPTED_X_TEST")
    if warnings and (len(warnings) != 0):
        print("======= WARNINGS =======")
        for warn in warnings:
            print(warn)


if __name__ == "__main__":
    main()
