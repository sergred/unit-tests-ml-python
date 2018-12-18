#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

""""""

from sklearn.model_selection import train_test_split as split
# from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
# from quilt.data.uciml import abalone
# from quilt.data.usr import credit
from tabulate import tabulate
import pandas as pd
import numpy as np
import unittest
import time
import os

from models import DecisionTree, RandomForest, ExtremelyRandomizedTrees, XGB
from models import LinearSVM, SVM, KNN, GausNB, BaggingRandomForest
from models import LogRegression, MLPC
from error_generation import ImplicitMissingValues, ExplicitMissingValues
from test_suite import AutomatedTestSuite, Test, Warning
from profilers import SklearnPipelineProfiler, DataFrameProfiler
from profilers import ErrorType, Severity
from pipelines import WineQualityPipeline, WineQualityMissingPipeline
from pipelines import AbalonePipeline, CreditGPipeline
from pipelines import AdultPipeline, AdultMissingPipeline
from pipelines import HeartPipeline
from error_generation import Anomalies, Typos
from analyzers import DataScale, DataType
from settings import get_resource_path
from messages import Message

np.random.seed(0)


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
            raise Exception('Wrong subs')

    def save(self, filename):
        np.savetxt(filename, self.table, delimiter=",", fmt="%s")

    def show(self):
        res = tabulate(self.table, tablefmt='psql')
        print(res)
        return res


class CreditGTest(unittest.TestCase):
    def setUp(self):
        self.resource_folder = get_resource_path()
        self.pipeline = CreditGPipeline()
        # data = credit.dataset_31_credit_g()
        data = pd.read_csv(os.path.join(self.resource_folder, 'data',
                                        'credit-g/dataset_31_credit-g.csv'))
        target = 'class'
        # I guess it will work only if the target value is the last one.
        self.features = [col for col in data.columns if col != target]
        X = data[self.features]
        y = data[target]
        sets = split(X, y, test_size=0.2, random_state=0)
        self.X_train, self.X_test, self.y_train, self.y_test = sets
        self.data_profile = DataFrameProfiler().on(self.X_train)
        self.automated_suite = AutomatedTestSuite()

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
        pipeline_profile = SklearnPipelineProfiler().on(model)
        tests, warnings = (self.automated_suite
                           .with_profiles(self.data_profile, pipeline_profile)
                           .on(corrupted_X_test))
        for column, profile in zip(columns, self.data_profile.profiles):
            self.assertIn(Test(Severity.CRITICAL).is_complete(profile), tests)
            self.assertIn(Warning(ErrorType.MISSING_VALUE,
                                  Severity.CRITICAL,
                                  Message().not_complete % column),
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
        pipeline_profile = SklearnPipelineProfiler().on(model)
        tests, warnings = (self.automated_suite
                           .with_profiles(self.data_profile, pipeline_profile)
                           .on(corrupted_X_test))
        for column, profile in zip(columns, self.data_profile.profiles):
            if profile.scale != DataScale.NOMINAL:
                continue
            self.assertIn(Test(Severity.CRITICAL).is_in_range(profile), tests)
            self.assertIn(Warning(ErrorType.NOT_IN_RANGE,
                                  Severity.CRITICAL, Message().not_in_range %
                                  (profile.column_name, str(profile.range))),
                          warnings)


class AbaloneTest(unittest.TestCase):
    def setUp(self):
        self.resource_folder = get_resource_path()
        self.pipeline = AbalonePipeline()
        # data = abalone.tables.abalone()
        data = pd.read_csv(
            os.path.join(self.resource_folder, 'data', 'abalone/abalone.csv'))
        target = 'Rings'
        # I guess it will work only if the target value is the last one.
        self.features = [col for col in data.columns if col != target]
        X = data[self.features]
        y = data[target]
        sets = split(X, y, test_size=0.2, random_state=0)
        self.X_train, self.X_test, self.y_train, self.y_test = sets
        self.data_profile = DataFrameProfiler().on(self.X_train)
        self.automated_suite = AutomatedTestSuite()

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
        pipeline_profile = SklearnPipelineProfiler().on(model)
        tests, warnings = (self.automated_suite
                           .with_profiles(self.data_profile, pipeline_profile)
                           .on(corrupted_X_test))
        for column, profile in zip(columns, self.data_profile.profiles):
            self.assertIn(Test(Severity.CRITICAL).is_complete(profile), tests)
            self.assertIn(Warning(ErrorType.MISSING_VALUE,
                                  Severity.CRITICAL,
                                  Message().not_complete % column),
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
        pipeline_profile = SklearnPipelineProfiler().on(model)
        tests, warnings = (self.automated_suite
                           .with_profiles(self.data_profile, pipeline_profile)
                           .on(corrupted_X_test))
        for column, profile in zip(columns, self.data_profile.profiles):
            if profile.scale != DataScale.NOMINAL:
                continue
            self.assertIn(Test(Severity.CRITICAL).is_in_range(profile), tests)
            self.assertIn(Warning(ErrorType.NOT_IN_RANGE,
                                  Severity.CRITICAL, Message().not_in_range %
                                  (profile.column_name, str(profile.range))),
                          warnings)

    # def test_(self):
    #     pass


# class Test(unittest.TestCase):
#     def setUp(self):
#         pass
#
#     def tearDown(self):
#         pass

class EvaluationSuite:
    def __init__(self):
        self.resource_folder = get_resource_path()
        # for dataset_name in sorted(os.listdir(folder)):
        #     if dataset_name.endswith('.csv'):
        #         print(dataset_name[:-4])
        self.pipelines = {
            'credit-g': (
                'credit-g/dataset_31_credit-g.csv', 'class',
                CreditGPipeline()),
            'wine-quality': (
                'wine-quality/wine-quality-red.csv', 'class',
                WineQualityPipeline()),
            'wq-missing': (
                'wine-quality/wine-quality-red.csv', 'class',
                WineQualityMissingPipeline()),
            'abalone': (
                'abalone/abalone.csv', 'Rings',
                AbalonePipeline()),
            'adult': (
                'adult/adult.csv', 'class',
                AdultPipeline()),
            'adult-missing': (
                'adult/adult.csv', 'class',
                AdultMissingPipeline()),
            'heart': (
                'heart/heart.csv', 'class',
                HeartPipeline())}

        self.classifiers = {'dtc': DecisionTree(),
                            'rfc40': RandomForest(size=40),
                            'ertc40': ExtremelyRandomizedTrees(size=40),
                            'xgb': XGB(),
                            'svm': SVM(),
                            'lsvm': LinearSVM(),
                            'knn': KNN(n_neighbors=7),
                            'logreg': LogRegression(),
                            'gaus': GausNB(),
                            'brfc40': BaggingRandomForest(size=40),
                            'mlpc': MLPC(input_size=[16, 32, 16, 8])
                            }

        self.error_gens = {
            'numeric anomalies': (
                Anomalies(), lambda x: x.dtype in [DataType.INTEGER,
                                                   DataType.FLOAT]),
            'typos': (
                Typos(), lambda x: x.dtype == DataType.STRING),
            'explicit misvals': (
                ExplicitMissingValues(), lambda x: True),
            'implicit misvals': (
                ImplicitMissingValues(), lambda x: True)}

        self.tests = {'num disc': lambda x: (x.scale == DataScale.NOMINAL
                                             and x.dtype in [DataType.INTEGER,
                                                             DataType.FLOAT]),
                      'num cont': lambda x: (x.scale == DataScale.NOMINAL
                                             and x.dtype in [DataType.INTEGER,
                                                             DataType.FLOAT]),
                      'string': lambda x: x.dtype == DataType.STRING}

        self.results = Table(rows=sorted(self.pipelines.keys()),
                             columns=sorted(self.classifiers.keys()),
                             subrows=self.tests.keys(),
                             subcolumns=self.error_gens.keys())

    def write_update(self, pipeline, classifier, err_gen,
                     column, column_type, result, step=""):
        with open('resources/results/log.csv', 'a') as f:
            f.write("%d;%s;%s;%s;%s;%s;%s;%s\n" % (
                int(time.time()), pipeline, classifier, err_gen, column,
                column_type, result, step))
        return self

    def run(self):
        # with self.assertRaises(SomeException) as cm:
        #     do_something()

        # the_exception = cm.exception
        # self.assertEqual(the_exception.error_code, 3)
        for pipe_idx, pipe in enumerate(sorted(self.pipelines.items())):
            pipeline_name, pipeline_content = pipe
            filename, target, pipeline = pipeline_content
            data = pd.read_csv(
                os.path.join(self.resource_folder, 'data', filename))
            print(pipeline_name)

            X = data[[col for col in data.columns if col != target]]
            y = data[target]
            X_train, X_test, y_train, y_test = split(X, y, test_size=0.2,
                                                     random_state=0)

            data_profiles = DataFrameProfiler().on(X_train).profiles

            # print(X_train.shape, X_test.shape)
            # for col in data.columns:
            #     if col == target:
            #         continue
            #     if (np.unique(X_train[col]).shape
            #             == np.unique(X_test[col]).shape):
            #         continue
            #     print(col)
            #     print(np.unique(X_train[col]))
            #     print(np.unique(X_test[col]))
            #     print

            list_of_classifiers = sorted(self.classifiers.items())
            for clf_idx, clf_content in enumerate(list_of_classifiers):
                cls_name, classifier = clf_content
                # print(cls_name)

                # for train_index, test_index in ss.split(X, y):
                #     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                #     y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                model = (pipeline
                         .with_estimator(classifier)
                         .fit(X_train, y_train))
                prediction = model.predict(X_test)
                res = accuracy_score(y_test, prediction)
                self.results.update("%.4f" % round(res, 4),
                                    pipe_idx, clf_idx)
                # print("%.4f" % round(res, 4))

                # print(pipeline.fit_transform(X_train, y_train).shape)

                list_of_err_gens = self.error_gens.items()
                for err_gen_idx, gen_content in enumerate(list_of_err_gens):
                    gen_name, info = gen_content
                    err_gen, gen_rule = info
                    list_of_tests = self.tests.items()
                    for type_idx, type_content in enumerate(list_of_tests):
                        type_name, type_rule = type_content
                        filtered_columns = [col.column_name
                                            for col in data_profiles
                                            if (type_rule(col)
                                                and gen_rule(col))]
                        print(type_name)
                        # print(filtered_columns)
                        if len(filtered_columns) == 0:
                            res = '---'
                        else:
                            count = 0
                            results = []
                            for col in filtered_columns:
                                try:
                                    # print(col)
                                    corrupted_X_test = err_gen.on(
                                        X_test, [col])
                                    steps = pipeline.pipe.steps
                                    # print(steps)
                                    for i in range(len(steps)):
                                        step_idx = i
                                        (Pipeline(steps[:i + 1])
                                         .fit(X_train, y_train)
                                         .transform(corrupted_X_test))
                                    step_idx = -1
                                    prediction = model.predict(
                                        corrupted_X_test)
                                    results.append(
                                        accuracy_score(y_test, prediction))
                                    self.write_update(pipeline_name,
                                                      cls_name, gen_name,
                                                      col, type_name,
                                                      "%.4f" % results[-1])
                                    count += 1
                                except Exception as exception:
                                    print("%s: %s" % (
                                        err_gen.__class__.__name__, exception))
                                    self.write_update(
                                        pipeline_name, cls_name, gen_name,
                                        col, type_name,
                                        "%s: %s" % (err_gen.__class__.__name__,
                                                    exception),
                                        steps[step_idx][0])

                            if len(results) == 0:
                                res = "(%d) FAIL" % (len(filtered_columns))
                            else:
                                res = "(%d/%d) %.4f +- %.4f" % (
                                    count, len(filtered_columns),
                                    np.mean(results), np.std(results))
                            print(res)

                        self.results.update(res, pipe_idx, clf_idx,
                                            type_idx, err_gen_idx)

        self.results.show()
        self.results.save(
            os.path.join(self.resource_folder, "results/matrix.csv"))


if __name__ == '__main__':
    EvaluationSuite().run()
    unittest.main()
