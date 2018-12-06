#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

""""""

from sklearn.model_selection import train_test_split as split
from sklearn.metrics import accuracy_score
from quilt.data.usr import credit, wine

import pandas as pd

from profilers import SklearnPipelineProfiler, DataFrameProfiler
from test_suite import AutomatedTestSuite, TestSuite, Test
from error_generation import ExplicitMissingValues
from pipelines import WineQualityPipeline
# from pipelines import CreditGPipeline
from models import RandomForest


def main():
    """
    """
    # data = credit.dataset_31_credit_g()
    data = wine.wine_quality_red_csv()
    print(data.shape)
    print(data.columns)

    target = "class"
    X, y = data[[col for col in data.columns if col != target]], data[target]
    X_train, X_test, y_train, y_test = split(X, y,
                                             test_size=0.2,
                                             random_state=0)

    # pipeline = CreditGPipeline()
    pipeline = WineQualityPipeline()
    classifier = RandomForest(size=40)
    model = pipeline.with_estimator(classifier).fit(X_train, y_train)

    prediction = model.predict(X_test)
    print(accuracy_score(y_test, prediction))

    suite = TestSuite()
    automated_suite = AutomatedTestSuite()
    data_profile = DataFrameProfiler().on(X_train)
    pipeline_profile = SklearnPipelineProfiler().on(model)

    suite.add(Test()
              .is_complete(data_profile.for_column('volatile_acidity'))
              .is_in_range(data_profile.for_column('alcohol')))

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

    tests, warnings = (automated_suite
                       .with_profiles(data_profile, pipeline_profile)
                       .run(corrupted_X_test))

    print("*** AUTOMATED_TEST_SUITE, CORRUPTED_X_TEST")
    if warnings and (len(warnings) != 0):
        print("======= WARNINGS =======")
        for warn in warnings:
            print(warn)


if __name__ == "__main__":
    main()
