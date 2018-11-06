#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

""""""

from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from error_generation import Anomalies, Typos, ExplicitMissingValues, ImplicitMissingValues
from tabulate import tabulate
import numpy as np
import openml

def main():
    """
    """
    # tasks = openml.tasks.list_tasks(task_type_id=1)
    # tasks = pd.DataFrame.from_dict(tasks, orient='index')
    # print(tasks.columns)

    # tasks = openml.tasks.list_tasks(tag='OpenML100')
    # tasks = pd.DataFrame.from_dict(tasks, orient='index')
    openml100 = [3, 6, 11, 12, 14, 15, 16, 18, 20, 21, 22, 23, 24, 28,
                 29, 31, 32, 36, 37, 41, 43, 45, 49, 53, 58, 219, 10093,
                 9914, 9946, 9950, 9964, 9967, 9968, 9970, 9971, 9976,
                 9977, 9978, 9979, 9980, 9981, 9983, 9985, 9986, 7592,
                 9952, 9954, 9955, 9956, 9957, 9960, 3510, 3512, 3543,
                 3549, 3021, 3022, 3481, 3485, 3492, 3493, 3494, 2074,
                 2079, 3560, 3561, 3567, 3573, 3902, 3903, 3904, 3913,
                 3917, 3918, 3946, 3948, 3889, 3891, 3896, 3899, 3954,
                 10101, 14964, 14965, 14966, 14967, 14968, 14969, 14970,
                 34536, 34537, 34538, 34539, 125920, 125921, 125922, 125923, 146195, 146606, 146607]

    # for i in range(13203, 9405080):
    #     try:
    #         if openml.runs.get_run(i).task_id in openml100:
    #             print(i)
    #             with open('openml_runs.txt', 'a') as f: f.write("%d\n" % i)
    #     except Exception as e:
    #         print("%d, %s" % (i, e))

    # benchmark_suite = openml.study.get_study('OpenML100', 'tasks') # obtain the benchmark suite
    # # build a sklearn classifier
    # clf = make_pipeline(SimpleImputer(), DecisionTreeClassifier())
    # for task_id in benchmark_suite.tasks: # iterate over all tasks
    #     task = openml.tasks.get_task(task_id) # download the OpenML task
    #     X, y = task.get_X_and_y() # get the data (not used in this example)
    #     print(', '.join("%s: %s" % item for item in vars(task).items()))
    #     # run classifier on splits (requires API key)
    #     run = openml.runs.run_model_on_task(task, clf)
    #     score = run.get_metric_score(accuracy_score) # print accuracy score
    #     print('Data set: %s; Accuracy: %0.2f' % (task.get_dataset().name,score.mean()))

    def evaluate(model, data, target):
        X, y = data[[col for col in data.columns if col != target]], data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        # print(X_train.shape)
        # results = []
        # ss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
        # for train_index, test_index in ss.split(X, y):
        #     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        #     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        try:
            new_model = model.fit(X_train, y_train)
            prediction = new_model.predict(X_test)
            res = "%.4f" % round(accuracy_score(y_test, prediction), 4)
        except Exception as e:
            print("%s: %s" % (err_gen.__class__, e))
            res = 'Fail'
        # results.append(res)

        # failed_count = len([item for item in results if item == 'Fail'])
        # results = [item for item in results if item != 'Fail']
        # tmp = "Failed: %d/%d, " % (failed_count, len(results)) if failed_count > 0 else ""
        # res = "%s%.4f +- %.4f" % (tmp, np.mean(results), np.std(results))
        return res

    error_gens = {'numeric anomalies': Anomalies(),
                  'typos': Typos(),
                  'explicit misvals': ExplicitMissingValues(),
                  'implicit misvals': ImplicitMissingValues()}

    task = openml.tasks.get_task(9900) # Supervised classification on Abalone dataset
    runs = openml.runs.list_runs(task=[task.task_id])
    results = np.zeros((len(runs)+1, len(error_gens)+2), dtype=object)
    results[0, 1:] = ['baseline'] + list(error_gens.keys())
    for run_idx, run in enumerate(runs):
        run = openml.runs.get_run(run)
        flow = openml.flows.get_flow(run.flow_id)
        # print(vars(flow))
        # if run.model is None: continue
        results[run_idx+1, 0] = run_idx
        perf_vals = list(run.fold_evaluations['predictive_accuracy'][0].values())
        results[run_idx+1, 1] = "%.4f +- %.4f" % (np.mean(perf_vals), np.std(perf_vals))
        # model = openml.runs.initialize_model_from_run(run.run_id)
        model = openml.flows.flow_to_sklearn(run.flow)
        # print(vars(model))
        print(model)
        data, columns = task.get_dataset().get_data(return_attribute_names=True)
        dataset = pd.DataFrame(data, columns=columns)
        # new_run = openml.runs.run_model_on_task(model, task)
        for err_gen_idx, err_gen in enumerate(error_gens.values()):
            results[run_idx+1, err_gen_idx+1] = evaluate(model, err_gen.on(dataset), task.target_name)

    print(tabulate(results, tablefmt='psql'))

if __name__ == "__main__":
    main()
