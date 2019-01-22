#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

""""""

from sklearn.model_selection import train_test_split as split
from sklearn.metrics import accuracy_score
from copy import deepcopy
import pandas as pd
import numpy as np
import itertools
import os

from models import DecisionTree, RandomForest, ExtremelyRandomizedTrees
from models import LinearSVM, SVM, KNN, GausNB, BaggingRandomForest
from models import LogRegression, MLPC, XGB
from error_generation import ImplicitMissingValues, ExplicitMissingValues
from error_generation import Anomalies, Typos, SwapFields
from profilers import DataFrameProfiler
from pipelines import AutomatedPipeline
from settings import get_resource_path

np.random.seed(0)


class MetaClassifier:
    def __init__(self, ml_pipeline, regressor):
        self.ml_pipeline = ml_pipeline
        self.regressor = regressor
        self.funcs = [np.min, np.max, np.mean, np.std,
                      self.quant25, self.quant50, self.quant75]

    def quant25(self, x): return np.quantile(x, 25)

    def quant50(self, x): return np.quantile(x, 50)

    def quant75(self, x): return np.quantile(x, 75)

    def stats(self, data):
        for column in data.T:
            for foo in self.funcs:
                yield foo(column)

    def train(self, X_train, y_train):
        features_X_train = [self.ml_pipeline.predict_proba(example)
                            for example in X_train]
        meta_X_train = np.array([self.stats(example)
                                 for example in features_X_train])
        meta_y_train = performance_metric(
            y_train, self.ml_pipeline.predict(X_train))
        self.regressor.fit(meta_X_train, meta_y_train)
        return self

    # def __getattr__(self, name):
    #     return getattr(self.regressor, name)


class Detector:
    def __init__(self):
        pass


class BlackBox:
    def __init__(self):
        self.model = None

    def train(self, clf, X_train, y_train):
        data_profile = DataFrameProfiler().on(X_train)
        pipeline = AutomatedPipeline(data_profile)
        self.model = pipeline.with_estimator(clf).fit(X_train, y_train)
        return self

    def __getattr__(self, name):
        return getattr(self.model, name)


class HyperParameterHolder:
    def __init__(self, hp):
        self.holder = self.unfold(hp)

    def __iter__(self):
        return self.holder

    def __next__(self):
        return next(self.holder)

    def unfold(self, hp):
        keys, values = tuple(zip(*hp.items()))
        values = [val if isinstance(val, list) else [val] for val in values]
        sizes = [list(range(len(val))) for val in values]
        for it in itertools.product(*sizes):
            yield dict([(keys[i], values[i][it[i]]) for i in range(len(keys))])


class ErrorGenerationStrategy:
    def __init__(self, error_generators, hyperparams):
        self.generators = error_generators
        self.hp = hyperparams

    def on(self, data, hp):
        return self.run(data, hp)

    def strategy(self, data, hp, rs):
        tmp = deepcopy(data)
        for i, idx in enumerate(np.argsort(hp['mask'])[::-1][:len(rs)]):
            np.random.seed(rs[i])
            tmp = self.generators[idx].on(tmp)
        return tmp

    def run(self, data, hp):
        factor = np.sum(np.array(hp['mask']) > 0)
        samples = factor * hp['testset_size']
        random_states = np.random.random_integers(1, 1000, samples)
        tmp = []
        for i in range(hp['testset_size']):
            tmp.append(self.strategy(data, hp,
                                     random_states[factor*i: factor*(i+1)]))
        np.random.seed(0)
        return tmp


def split_dataset(data, target_feature, hp):
    train, val = hp['train_ratio'], hp['val_ratio']
    test, target = hp['test_ratio'], hp['target_ratio']
    denominator = np.sum([train, val, test, target])
    train /= denominator
    val /= denominator
    test /= denominator
    target /= denominator
    random_state = hp['random_state']
    X = data[[col for col in data.columns if col != target_feature]]
    y = data[target_feature]
    X_rest, X_train, y_rest, y_train = split(X, y,
                                             test_size=train,
                                             random_state=random_state)
    X_rest, X_target, y_rest, y_target = split(X_rest, y_rest,
                                               test_size=target/(1.-train),
                                               random_state=random_state)
    X_val, X_test, y_val, y_test = split(X_rest, y_rest,
                                         test_size=test/(val+test),
                                         random_state=random_state)
    return X_train, y_train, X_val, y_val, X_test, y_test, X_target, y_target


def performance_metric(*_):
    return accuracy_score(*_)


def main():
    """
    """
    path = get_resource_path()

    classifiers = [
        DecisionTree(),
        RandomForest(size=40),
        ExtremelyRandomizedTrees(size=40),
        XGB(),
        SVM(),
        LinearSVM(),
        KNN(n_neighbors=7),
        LogRegression(),
        GausNB(),
        BaggingRandomForest(size=40),
        MLPC(input_size=[16, 32, 16, 8])
    ]

    error_generators = [
        Anomalies(),
        Typos(),
        ExplicitMissingValues(),
        ImplicitMissingValues(),
        SwapFields()
    ]

    # TODO: dataset size as a hyperparameter
    # TODO: random_state as a hyperparameter
    hyperparams = {
        'train_ratio': .7,
        'val_ratio': .1,
        'test_ratio': .1,
        'target_ratio': .1,
        'random_state': [0],
        'row_fraction': [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8],
        'classifier': classifiers,
        # Ordering of error generators
        'mask': [(0, 0, 1, 0, 0), (0, 0, 0, 1, 0), (0, 0, 0, 0, 1),
                 (0, 2, 0, 0, 1)],
        'testset_size': 1000
    }

    datasets = pd.read_csv(os.path.join(path, 'datasets.csv'))

    for dataset_info in datasets.values:
        filepath, name, target_feature, task = tuple(dataset_info)
        data = pd.read_csv(os.path.join(path, 'data', filepath))

        for state in HyperParameterHolder(hyperparams):
            # Dataset Split
            (X_train, y_train, X_val, y_val,
             X_test, y_test, X_target, y_target) = split_dataset(
                 data, target_feature, state)

            tuning_done = False
            while not tuning_done:
                # ML Pipeline Training Procedure
                model = BlackBox().train(state['classifier'], X_train, y_train)

                # ML Pipeline Validation Procedures
                predicted = model.predict(X_val)
                score = performance_metric(y_val, predicted)
                print("%.4f" % round(score, 4))
                tuning_done = True

            # ML Pipeline final performance score
            predicted = model.predict(X_test)
            score = performance_metric(y_test, predicted)
            print("ML Pipeline final perf score: %.4f" % round(score, 4))

            # Meta Classifier Training Procedure
            error_gen_strat = ErrorGenerationStrategy(error_generators, state)
            # TODO: so far, X_test/y_test is used for training

            # prepare a dataset based on X_test and repeated error generation
            features_X_test = error_gen_strat.on(X_test, state)

            meta_classifier = MetaClassifier(model, GausNB())
            meta_classifier.train(features_X_test, y_test)

            # Meta Classifier Evaluation Procedure
            X_eval = error_gen_strat.on(X_target)
            predicted = meta_classifier.predict(X_eval)
            results = performance_metric(y_target, predicted)

            print(results)


if __name__ == "__main__":
    main()
