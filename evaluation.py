#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

""""""

from sklearn.model_selection import train_test_split as split
from quilt.data.usr import credit
from tabulate import tabulate
import pandas as pd
import numpy as np
import unittest

from models import DecisionTree, RandomForest, ExtremelyRandomizedTrees, XGB
from error_generation import ImplicitMissingValues, ExplicitMissingValues
from profilers import SklearnPipelineProfiler, ErrorType, Severity
from pipelines import WineQualityPipeline, WineQualityMissingPipeline
from pipelines import AbalonePipeline, CreditGPipeline
from test_suite import AutomatedTestSuite, TestSuite, Test, Warning
from error_generation import Anomalies, Typos


class Table:
    def __init__(self, rows, columns, subrows, subcolumns):
        self.row_header_len, self.col_header_len = 2, 2
        self.rows_len, self.cols_len = len(rows), len(columns)
        self.subrows_len, self.subcols_len = len(subrows) + 1, len(subcolumns)
        self.dataset_baseline_title = 'baseline'
        self.table = np.zeros(
            (self.rows_len*self.subrows_len+self.row_header_len,
             self.cols_len*self.subcols_len+self.col_header_len), dtype=object)
        self.table[:self.col_header_len, :self.row_header_len] = np.array(
            [['', 'classifiers'], ['datasets', 'tests\\anomalies']])
        self.table[0, self.row_header_len:] = np.array(
            [list(columns)[i//self.subcols_len]
             for i in range(self.cols_len*self.subcols_len)])
        self.table[1, self.row_header_len:] = np.array(
            list(subcolumns)*self.cols_len)
        self.table[self.col_header_len:, 0] = np.array(
            [list(rows)[i//self.subrows_len]
             if i % self.subrows_len == 0 else ''
             for i in range(self.rows_len*self.subrows_len)])
        self.table[self.col_header_len:, 1] = np.array(
            ([self.dataset_baseline_title] + list(subrows))*self.rows_len)

    def update(self, data, i, j, x=-1, y=-1):
        if x == -1 and y == -1:
            z = self.row_header_len + self.subcols_len*j
            self.table[self.col_header_len + self.subrows_len*i,
                       z:z+self.subcols_len] = data
        elif x != -1 and y != -1:
            self.table[self.col_header_len + self.subrows_len*i + x + 1,
                       self.row_header_len + self.subcols_len*j + y] = data
        else:
            raise Exception('wrong subs')

    def save(self, filename):
        np.savetxt(filename, self.table, delimiter=",", fmt="%s")

    def show(self):
        res = tabulate(self.table, tablefmt='psql')
        print(res)
        return res


class EvaluationSuite(unittest.TestSuite):
    def __init__(self):
        pass

    def run(self):
        # with self.assertRaises(SomeException) as cm:
        #     do_something()

        # the_exception = cm.exception
        # self.assertEqual(the_exception.error_code, 3)
        pass


class CreditGTest(unittest.TestCase):
    def setUp(self):
        self.pipeline = CreditGPipeline()
        data = credit.dataset_31_credit_g()
        target = "class"
        # I guess it will work only if the target value is the last one.
        self.features = [col for col in data.columns if col != target]
        X = data[self.features]
        y = data[target]
        sets = split(X, y, test_size=0.2, random_state=0)
        self.X_train, self.X_test, self.y_train, self.y_test = sets

    def tearDown(self):
        pass

    def test_missing_values_in_data_with_random_forest(self):
        classifier = RandomForest()
        model = self.pipeline.with_estimator(classifier).fit(self.X_train,
                                                             self.y_train)
        error_generator = ExplicitMissingValues()
        num_cols = 3
        # Will fail if the number of columns is less than 3
        columns = np.random.choice(self.features, num_cols, replace=False)
        corrupted_X_test = error_generator.run(self.X_test, columns=columns)
        # prediction = model.predict(X_test)
        # print(accuracy_score(y_test, prediction))

        # suite = TestSuite()
        automated_suite = AutomatedTestSuite()
        pipeline_profile = SklearnPipelineProfiler().on(model)
        tests, warnings = automated_suite.run(corrupted_X_test,
                                              pipeline_profile)
        for column in columns:
            self.assertIn(Test(Severity.CRITICAL).is_complete(column), tests)
            self.assertIn(Warning(ErrorType.MISSING_VALUE,
                                  Severity.CRITICAL, """
                                  Column %s is not complete""" % column),
                          warnings)

    def test_typos_in_data_with_random_forest(self):
        classifier = RandomForest()
        model = self.pipeline.with_estimator(classifier).fit(self.X_train,
                                                             self.y_train)
        error_generator = Typos()
        num_cols = 3
        # Will fail if the number of columns is less than 3
        columns = np.random.choice(self.features, num_cols, replace=False)
        corrupted_X_test = error_generator.run(self.X_test, columns=columns)
        # prediction = model.predict(X_test)
        # print(accuracy_score(y_test, prediction))

        # suite = TestSuite()
        automated_suite = AutomatedTestSuite()
        pipeline_profile = SklearnPipelineProfiler().on(model)
        tests, warnings = automated_suite.run(corrupted_X_test,
                                              pipeline_profile)
        for column in columns:
            self.assertIn(Test(Severity.CRITICAL).is_in_range(column), tests)
            self.assertIn(Warning(ErrorType.TYPO,
                                  Severity.CRITICAL, """
                                  Column %s is not complete""" % column),
                          warnings)

    # def test_(self):
    #     pass


# class Test(unittest.TestCase):
#     def setUp(self):
#         pass
#
#     def tearDown(self):
#         pass


if __name__ == '__main__':
    unittest.main()
