import hilda.datasets as datasets
import hilda.perturbations as pertubations
import hilda.learners as learners
import hilda.experiments as exp
import numpy as np

import warnings
warnings.simplefilter("ignore")


def outlier_perturbations(dataset):
    perturbations = []
    for num_columns_affected in range(1, len(dataset.numerical_columns)):
        for fraction_of_outliers in [0.0, 0.05, 0.25, 0.5, 0.75, 0.99]:
            for _ in range(0, 100):
                columns_affected = np.random.choice(dataset.numerical_columns, num_columns_affected)
                perturbations.append(pertubations.Outliers(fraction_of_outliers, columns_affected))

    return perturbations


def run_one(dataset, learner):
    exp.reapply_perturbations(dataset, learner, outlier_perturbations(dataset), outlier_perturbations(dataset),
                              'hilda_outliers')


run_one(datasets.BalancedAdultDataset(), learners.LogisticRegression('accuracy'))
run_one(datasets.BalancedAdultDataset(), learners.DNN('accuracy'))
run_one(datasets.AdultDataset(), learners.LogisticRegression('roc_auc'))
run_one(datasets.AdultDataset(), learners.DNN('roc_auc'))
run_one(datasets.CardioDataset(), learners.LogisticRegression('accuracy'))
run_one(datasets.CardioDataset(), learners.DNN('accuracy'))
run_one(datasets.CardioDataset(), learners.LogisticRegression('roc_auc'))
run_one(datasets.CardioDataset(), learners.DNN('roc_auc'))
