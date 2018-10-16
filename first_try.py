#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import os

from models import DecisionTree, RandomForest, ExtremelyRandomizedTrees, XGB
from pipelines import AbalonePipeline, CreditGPipeline, WineQualityPipeline
from error_generation import MissingValues, Anomalies
from settings import get_resource_path

np.random.seed(0)

def main():
    resource_folder = get_resource_path()
    # for dataset_name in sorted(os.listdir(folder)):
    #     if dataset_name.endswith('.csv'):
    #         print(dataset_name[:-4])

    pipelines = {'credit-g': ('dataset_31_credit-g.csv', 'class', CreditGPipeline()),
                 'wine-quality': ('wine-quality-red.csv', 'class', WineQualityPipeline())}
    #            'abalone': ('abalone.csv', 'Rings', AbalonePipeline),

    classifiers = {'dtc': DecisionTree(),
                   'rfc': RandomForest(size=40),
                   'ertc': ExtremelyRandomizedTrees(size=40),
                   'xgb': XGB()}

    error_gens = {'anomalies': Anomalies(),
                  'missing values': MissingValues()}

    results = np.zeros((2*len(pipelines)+2, len(error_gens)*len(classifiers)+1), dtype=object)
    data_names = sorted(pipelines.keys())
    results[0, :] = np.array([""] + [list(classifiers)[i//len(error_gens)]for i in range(len(error_gens)*len(classifiers))])
    results[1, :] = np.array([""] + list(error_gens)*len(classifiers))
    results[:, 0] = np.array(["", ""] + [item for sublist in map(lambda x: [x, ""], data_names) for item in sublist])

    for pipe_idx, pipe in enumerate(sorted(pipelines.items())):
        name, content = pipe
        filename, target, pipeline = content
        data = pd.read_csv(os.path.join(resource_folder, "data", filename))
        # print(name)
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

            i = len(error_gens)*classifier_idx + 1
            results[2*pipe_idx + 2, i:i+len(error_gens)] = accuracy_score(y_test, prediction)

            for err_gen_idx, err_gen in enumerate(error_gens):

                model = pipeline.with_estimator(classifier).fit(X_train, y_train)
                prediction = model.predict(X_test)

                results[2*pipe_idx + 3, len(error_gens)*classifier_idx + err_gen_idx + 1] = accuracy_score(y_test, prediction)

    np.savetxt(os.path.join(resource_folder, "results/matrix.csv"), results, delimiter=",", fmt="%s")


if __name__ == "__main__":
    main()
