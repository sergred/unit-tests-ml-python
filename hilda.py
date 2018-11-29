#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

""""""

from sklearn.model_selection import train_test_split as split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

from pipelines import CreditGPipeline, WineQualityMissingPipeline
from profilers import DataFrameProfiler, SklearnPipelineProfiler
from test_suite import AutomatedTestSuite, TestSuite, Test
from error_generation import ExplicitMissingValues
from models import RandomForest

def main():
    """
    """
    # data = pd.read_csv('resources/data/dataset_31_credit-g.csv')
    data = pd.read_csv('resources/data/wine-quality-red.csv')
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

    suite.add(Test()
              .is_complete('volatile_acidity'))

    warnings = suite.on(X_test).run()

    if warnings and (len(warnings) != 0):
        print("======= WARNINGS =======")
        for warn in warnings:
            print(warn)

    error_generator = ExplicitMissingValues()
    corrupted_X_test = error_generator.run(X_test, ['volatile_acidity'])

    warnings = suite.on(corrupted_X_test).run()

    if warnings and (len(warnings) != 0):
        print("======= WARNINGS =======")
        for warn in warnings:
            print(warn)

    print()

    data_profile = DataFrameProfiler().on(X_train)
    pipeline_profile = SklearnPipelineProfiler().on(model)
    automated_suite = AutomatedTestSuite(data_profile, pipeline_profile)
    tests, warnings = automated_suite.run()

    if warnings and (len(warnings) != 0):
        print("======= WARNINGS =======")
        for warn in warnings:
            print(warn)



if __name__ == "__main__":
    main()
