import hilda.datasets as datasets
import hilda.perturbations as pertubations
import hilda.learners as learners
import numpy as np
import hilda.experiments as exp

import warnings
warnings.simplefilter("ignore")

#dataset = datasets.BalancedAdultDataset()
dataset = datasets.CardioDataset()

perturbations_for_training = []
for num_columns_affected in range(1, len(dataset.numerical_columns)):
    for fraction_of_outliers in [0.0, 0.05, 0.25, 0.5, 0.75, 0.99]:
        for _ in range(0, 1):
            columns_affected = np.random.choice(dataset.numerical_columns, num_columns_affected)
            perturbations_for_training.append(pertubations.Outliers(fraction_of_outliers, columns_affected))

# generate a bunch of perturbations for evaluation
perturbations_for_evaluation = []
for num_columns_affected in range(1, len(dataset.numerical_columns)):
    for fraction_of_outliers in [0.0, 0.05, 0.25, 0.5, 0.75, 0.99]:
        for _ in range(0, 1):
            columns_affected = np.random.choice(dataset.numerical_columns, num_columns_affected)
            perturbations_for_evaluation.append(pertubations.Outliers(fraction_of_outliers, columns_affected))


# name the perturbations
perturbations_name = "calibrations_outlier"

# define the learner
learner = learners.DNN('accuracy')
#learner = learners.DNN('roc_auc')
#learner = learners.LogisticRegression('roc_auc')
#learner = learners.LogisticRegression('accuracy')

#learner = learners.XgBoost('accuracy')

# run an experiment
exp.compare_to_calibration(dataset, learner, perturbations_for_training,
                                     perturbations_for_evaluation, perturbations_name)

