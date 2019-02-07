import hilda.datasets as datasets
import hilda.perturbations as pertubations
import hilda.learners as learners
import hilda.experiments as exp
import numpy as np

import warnings
warnings.simplefilter("ignore")


def missing_perturbations(dataset, substitute):
    perturbations = []
    for num_columns_affected in range(1, len(dataset.categorical_columns)):
        for fraction_of_values_to_delete in [0.0, 0.05, 0.25, 0.5, 0.75, 0.99]:
            for _ in range(0, 100):
                columns_affected = np.random.choice(dataset.categorical_columns, num_columns_affected)
                perturbations.append(pertubations.MissingValues(fraction_of_values_to_delete,
                                     columns_affected, substitute))

    return perturbations


def run_one(dataset, learner, substitute):
    exp.reapply_perturbations(dataset, learner, missing_perturbations(dataset, substitute),
                              missing_perturbations(dataset, substitute), 'hilda_missing_values')


run_one(datasets.BalancedAdultDataset(), learners.LogisticRegression('accuracy'), 'n/a')
run_one(datasets.BalancedAdultDataset(), learners.DNN('accuracy'), 'n/a')
run_one(datasets.AdultDataset(), learners.LogisticRegression('roc_auc'), 'n/a')
run_one(datasets.AdultDataset(), learners.DNN('roc_auc'), 'n/a')
run_one(datasets.CardioDataset(), learners.LogisticRegression('accuracy'), -1)
run_one(datasets.CardioDataset(), learners.DNN('accuracy'), -1)
run_one(datasets.CardioDataset(), learners.LogisticRegression('roc_auc'), -1)
run_one(datasets.CardioDataset(), learners.DNN('roc_auc'), -1)