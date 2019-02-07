import hilda.datasets as datasets
import hilda.perturbations as pertubations
import hilda.learners as learners
import hilda.experiments as exp

import warnings
warnings.simplefilter("ignore")


def leet_perturbations():
    perturbations = []
    for fraction_of_troll_tweets in [0.0, 0.05, 0.25, 0.5, 0.75, 0.99]:
        for _ in range(0, 100):
            perturbations.append(pertubations.Leetspeak(fraction_of_troll_tweets, 'content', 'label', 1))

    return perturbations


def run_one(dataset, learner):
    exp.reapply_perturbations(dataset, learner, leet_perturbations(), leet_perturbations(), 'hilda_adversarial')


run_one(datasets.BalancedTrollingDataset(), learners.LogisticRegression('accuracy'))
run_one(datasets.BalancedTrollingDataset(), learners.DNN('accuracy'))
run_one(datasets.TrollingDataset(), learners.LogisticRegression('roc_auc'))
run_one(datasets.TrollingDataset(), learners.DNN('roc_auc'))
