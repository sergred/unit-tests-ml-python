#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

""""""

from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import openml

def main():
    """
    """
    openml100 = [openml.tasks.get_task(t).dataset_id for t in openml.study.get_study('OpenML100', 'tasks').tasks]
    runs = [i for i in range(9405080) if openml.runs.get_run(i).dataset_id in openml100]
    print(len(runs))
    with open('openml_runs.txt', 'w') as f:
        for item in runs: f.write("%s\n" % item)
        
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


if __name__ == "__main__":
    main()
