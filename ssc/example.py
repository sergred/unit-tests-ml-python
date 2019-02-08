import hilda.datasets as datasets
import hilda.perturbations as pertubations
import hilda.learners as learners
import numpy as np
import hilda.experiments as exp

import warnings
warnings.simplefilter("ignore")

# Pick a dataset
#dataset = datasets.CardioDataset()
dataset = datasets.BalancedAdultDataset()
#dataset = datasets.AdultDataset()
#dataset = datasets.BalancedTrollingDataset()
#dataset = datasets.TrollingDataset()

# # generate a bunch of perturbations for training
# perturbations_for_training = []
# for fraction_of_troll_tweets in [0.0, 0.05, 0.25, 0.5, 0.75, 0.99]:
#     for _ in range(0, 100):
#         perturbations_for_training.append(pertubations.Leetspeak(fraction_of_troll_tweets, 'content', 'label', 1))
#
# # generate a bunch of perturbations for evaluation
# perturbations_for_evaluation = []
# for fraction_of_troll_tweets in [0.0, 0.05, 0.25, 0.5, 0.75, 0.99]:
#     for _ in range(0, 100):
#         perturbations_for_evaluation.append(pertubations.Leetspeak(fraction_of_troll_tweets, 'content', 'label', 1))


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
            perturbations_for_evaluation.append(pertubations.MissingValues(fraction_of_values_to_delete,
                                                columns_affected, -1))

# perturbations_for_training = []
# for num_columns_affected in range(1, len(dataset.numerical_columns)):
#     for fraction_of_outliers in [0.0, 0.05, 0.25, 0.5, 0.75, 0.99]:
#         for _ in range(0, 1):
#             columns_affected = np.random.choice(dataset.numerical_columns, num_columns_affected)
#             perturbations_for_training.append(pertubations.Outliers(fraction_of_outliers, columns_affected))
#
# # generate a bunch of perturbations for evaluation
# perturbations_for_evaluation = []
# for num_columns_affected in range(1, len(dataset.numerical_columns)):
#     for fraction_of_outliers in [0.0, 0.05, 0.25, 0.5, 0.75, 0.99]:
#         for _ in range(0, 1):
#             columns_affected = np.random.choice(dataset.numerical_columns, num_columns_affected)
#             perturbations_for_evaluation.append(pertubations.Outliers(fraction_of_outliers, columns_affected))

# name the perturbations
perturbations_name = "example"

# define the learner
#learner = learners.DNN('accuracy')
#learner = learners.DNN('roc_auc')
#learner = learners.LogisticRegression('roc_auc')
learner = learners.LogisticRegression('accuracy')

#learner = learners.XgBoost('accuracy')

# run an experiment
log_line = exp.reapply_perturbations(dataset, learner, perturbations_for_training,
                                     perturbations_for_evaluation, perturbations_name)

print("---------------------------------------------------------------------------------------------------------------")
print(log_line)
