import hilda.datasets as datasets
import hilda.perturbations as pertubations
import hilda.learners as learners
import numpy as np
import hilda.experiments as exp

import warnings
warnings.simplefilter("ignore")

# Pick a dataset
dataset = datasets.CardioDataset()
# dataset = datasets.AdultDataset()
# dataset = datasets.BankmarketingDataset()

# generate a bunch of perturbations for training
perturbations_for_training = []
for num_columns_affected in range(1, 5):
    for fraction_of_values_to_delete in [0.0, 0.05, 0.25, 0.5, 0.75, 0.99]:
        for _ in range(0, 100):
            columns_affected = np.random.choice(dataset.categorical_columns, num_columns_affected)
            perturbations_for_training.append(pertubations.MissingValues(fraction_of_values_to_delete, columns_affected, -1))

# generate a bunch of perturbations for evaluation
perturbations_for_evaluation = []
for num_columns_affected in range(1, 5):
    for fraction_of_values_to_delete in [0.0, 0.05, 0.25, 0.5, 0.75, 0.99]:
        for _ in range(0, 100):
            columns_affected = np.random.choice(dataset.categorical_columns, num_columns_affected)
            perturbations_for_evaluation.append(pertubations.MissingValues(fraction_of_values_to_delete, columns_affected, -1))

perturbations_name = "missing_values_at_random"

# define the learner
# learn_with = learners.learn_logistic_regression_model
# learner_name = "logistic_regression"

learn_with = learners.learn_feed_forward_network
learner_name = "dnn"

# run an experiment
log_line = exp.reapply_perturbations(dataset, learn_with, learner_name, perturbations_for_training,
                                     perturbations_for_evaluation, perturbations_name)

print("---------------------------------------------------------------------------------------------------------------")
print(log_line)
