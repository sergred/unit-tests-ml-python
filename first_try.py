#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

# from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import os

from models import DecisionTree, RandomForest, ExtremelyRandomizedTrees, XGB
from pipelines import AbalonePipeline, CreditGPipeline
from pipelines import WineQualityPipeline, WineQualityMissingPipeline
from error_generation import ImplicitMissingValues, ExplicitMissingValues
from error_generation import Anomalies, Typos
from settings import get_resource_path
from evaluation import Table

np.random.seed(0)


def main():
    resource_folder = get_resource_path()
    # for dataset_name in sorted(os.listdir(folder)):
    #     if dataset_name.endswith('.csv'):
    #         print(dataset_name[:-4])

    pipelines = {'credit-g': ('credit-g/dataset_31_credit-g.csv',
                              'class', CreditGPipeline()),
                 'wine-quality': ('wine-quality/wine-quality-red.csv',
                                  'class', WineQualityPipeline()),
                 'wq-missing': ('wine-quality/wine-quality-red.csv',
                                'class', WineQualityMissingPipeline()),
                 'abalone': ('abalone/abalone.csv',
                             'Rings', AbalonePipeline())}

    classifiers = {'dtc': DecisionTree(),
                   'rfc': RandomForest(size=40),
                   'ertc': ExtremelyRandomizedTrees(size=40),
                   'xgb': XGB()}

    error_gens = {'numeric anomalies': (Anomalies(), 'numeric'),
                  'typos': (Typos(), 'string'),
                  'explicit misvals': (ExplicitMissingValues(), None),
                  'implicit misvals': (ImplicitMissingValues(), None)}

    tests = {'core': "mock"}

    results = Table(rows=sorted(pipelines.keys()),
                    columns=sorted(classifiers.keys()),
                    subrows=tests.keys(),
                    subcolumns=error_gens.keys())

    for pipe_idx, pipe in enumerate(sorted(pipelines.items())):
        name, pipeline_content = pipe
        filename, target, pipeline = pipeline_content
        data = pd.read_csv(os.path.join(resource_folder, "data", filename))
        print(name)

        X = data[[col for col in data.columns if col != target]]
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.2,
                                                            random_state=0)

        for clf_idx, clf_content in enumerate(sorted(classifiers.items())):
            cls_name, classifier = clf_content
            # print(cls_name)

            # for train_index, test_index in ss.split(X, y):
            #     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            #     y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model = pipeline.with_estimator(classifier).fit(X_train, y_train)
            prediction = model.predict(X_test)
            base_score = accuracy_score(y_test, prediction)
            results.update("%.4f" % round(base_score, 4),
                           pipe_idx, clf_idx)

            # print(pipeline.fit_transform(X_train, y_train).shape)

            for err_gen_idx, gen_content in enumerate(error_gens.values()):
                err_gen, cols = gen_content
                # corrupted_X_test = err_gen.on(X_test, cols)
                # model = pipeline.with_estimator(classifier).fit(X_train,
                #                                                 y_train)
                # prediction = model.predict(corrupted_X_test)
                # accuracy = accuracy_score(y_test, prediction)
                # res = "%.4f" % round(accuracy - base_score, 4)
                try:
                    corrupted_X_test = err_gen.on(X_test, cols)
                    model = pipeline.with_estimator(classifier).fit(X_train,
                                                                    y_train)
                    prediction = model.predict(corrupted_X_test)
                    accuracy = accuracy_score(y_test, prediction)
                    res = "%.4f" % round(accuracy - base_score, 4)
                except Exception as e:
                    print("%s: %s" % (err_gen.__class__, e))
                    res = 'Fail'

                results.update(res, pipe_idx, clf_idx, 0, err_gen_idx)

    results.show()
    results.save(os.path.join(resource_folder, "results/matrix.csv"))


if __name__ == "__main__":
    main()
