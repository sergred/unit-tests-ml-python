#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

""""""

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split as split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from copy import deepcopy
import pandas as pd
import numpy as np
import itertools
import os

from models import DecisionTree, RandomForest, ExtremelyRandomizedTrees
from models import LinearSVM, SVM, KNN, GausNB, BaggingRandomForest
from models import LinearRegression, LogRegression, MLPC, XGB
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

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return ("MetaClassifier(Model(%s), Regressor(%s))" %
                (self.ml_pipeline.steps[-1][-1].name, self.regressor.name))

    def quant25(self, x): return np.percentile(x, 25)

    def quant50(self, x): return np.percentile(x, 50)

    def quant75(self, x): return np.percentile(x, 75)

    def stats(self, data):
        for column in data.T:
            for foo in self.funcs:
                yield foo(column)

    def transform(self, X):
        features_X = [self.ml_pipeline.predict_proba(example) for example in X]
        return [list(self.stats(item)) for item in features_X]

    def fit(self, X_train, y_train):
        meta_y_train = [
            performance_metric(y_train, self.ml_pipeline.predict(x))
            for x in X_train]
        self.regressor.fit(self.transform(X_train), meta_y_train)
        return self

    def predict(self, X_test):
        return self.regressor.predict(self.transform(X_test))

    # def __getattr__(self, name):
    #     return getattr(self.regressor, name)


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

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        factor = np.sum(np.array(self.hp['mask']) > 0)
        gens = [self.generators[i].name
                for i in np.argsort(self.hp['mask'])[::-1][:factor]]
        return "ErrorGenStrategy(%s)" % (", ".join(gens),)

    def on(self, data, hp):
        return self.run(data, hp)

    def strategy(self, data, hp, rs):
        tmp = deepcopy(data)
        for i, idx in enumerate(np.argsort(hp['mask'])[::-1][:len(rs)]):
            np.random.seed(rs[i])
            tmp = self.generators[idx].on(tmp, row_fraction=hp['row_fraction'])
        np.random.seed(0)
        return tmp

    def run(self, data, hp):
        number_of_err_gens = np.sum(np.array(hp['mask']) > 0)
        number_of_samples = number_of_err_gens * hp['testset_size']
        random_states = np.random.random_integers(1, 1000, number_of_samples)
        return [self.strategy(data, hp,
                              random_states[number_of_err_gens*i:
                                            number_of_err_gens*(i+1)])
                for i in range(hp['testset_size'])]


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


def distance_metric(X_actual, X_predicted):
    return mean_squared_error(X_actual, X_predicted)


def main():
    """
    """
    path = get_resource_path()

    classifiers = [
        # DecisionTree(),
        # RandomForest(size=40),
        # ExtremelyRandomizedTrees(size=40),
        # XGB(),
        # SVM(),
        # LinearSVM(),
        # KNN(n_neighbors=7),
        LogRegression(),
        # GausNB(),
        # BaggingRandomForest(size=40),
        # MLPC(input_size=[16, 32, 16, 8])
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
        # 'row_fraction': [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8],
        'row_fraction': [0.2],
        'classifier': classifiers,
        # Ordering of error generators
        # 'mask': [(0, 0, 1, 0, 0), (0, 0, 0, 1, 0), (0, 0, 0, 0, 1),
        #          (0, 2, 0, 0, 1)],
        'mask': [(0, 0, 0, 1, 0)],
        'testset_size': 100
    }

    datasets = pd.read_csv(os.path.join(path, 'datasets.csv'))

    for dataset_info in datasets.values:
        filepath, name, target_feature, task = tuple(dataset_info)
        data = pd.read_csv(os.path.join(path, 'data', filepath))

        for state in HyperParameterHolder(hyperparams):
            print("HyperParam : %s" % str(state))
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
                print("Validation : accuracy = %.4f" % round(score, 4))
                tuning_done = True

            # ML Pipeline final performance score
            predicted = model.predict(X_test)
            score = performance_metric(y_test, predicted)
            print("Test       : accuracy = %.4f" % round(score, 4))

            # Meta Classifier Training Procedure
            error_gen_strat = ErrorGenerationStrategy(error_generators, state)
            # TODO: so far, X_test/y_test is used for training

            # prepare a dataset based on X_test and repeated error generation
            # NB: returns a python list, not a numpy array or pandas dataframe
            list_of_corrupted_X_test = error_gen_strat.on(X_test, state)

            try:
                meta_classifier = MetaClassifier(model, LinearRegression())
                print(str(meta_classifier))
                meta_classifier.fit(list_of_corrupted_X_test, y_test)

                # Meta Classifier Evaluation Procedure
                list_of_corrupted_X_target = error_gen_strat.on(
                    X_target, state)
                predicted_scores = meta_classifier.predict(
                    list_of_corrupted_X_target)
                actual_scores = [performance_metric(y_target, model.predict(x))
                                 for x in list_of_corrupted_X_target]
                plt.plot(range(len(actual_scores)), actual_scores, 'g^')
                plt.plot(range(len(predicted_scores)), predicted_scores, 'ro')
                plt.gca().legend(('ground truth', 'predicted scores'))
                plt.grid(True)
                plt.show()
                result = distance_metric(actual_scores, predicted_scores)

                print("Evaluation : distance metric = %.4f" % round(result, 4))
                print()
            except Exception as e:
                print("\nException  : %s\n%s\n" % (str(error_gen_strat), e))


if __name__ == "__main__":
    main()
