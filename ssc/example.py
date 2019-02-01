import hilda.datasets as datasets
import hilda.perturbations as pertubations
import hilda.learners as learners
import numpy as np
import hilda.experiments as exp

import warnings
warnings.simplefilter("ignore")

# Pick a dataset
dataset = datasets.AdultDataset()

# generate a bunch of perturbations
perturbations_to_apply = []
for num_columns_affected in range(1, 5):
    for fraction_of_values_to_delete in [0.0, 0.05, 0.25, 0.5, 0.75, 0.99]:
        for _ in range(0, 1):
            columns_affected = np.random.choice(dataset.categorical_columns, num_columns_affected)
            perturbations_to_apply.append(pertubations.MissingValues(fraction_of_values_to_delete, columns_affected))

perturbations_name = "missing_values_at_random"


log_line = exp.reapply_perturbations(dataset, learners.learn_logistic_regression, "logistic_regression",
                                       perturbations_to_apply, perturbations_name)

print("---------------------------------------------------------------------------------------------------------------")
print(log_line)
