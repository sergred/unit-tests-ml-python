#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

""""""

import numpy as np
from ssc.hilda.meta_regressors import train_random_forest_regressor


# Parametrized decorator that takes X_test and X_target
# to build and evaluate meta regressor
def validate_on(X_test, y_test, X_target, y_target):
    def percentiles_of_probas(predictions):
        probs_class_a = np.transpose(predictions)[0]
        probs_class_b = np.transpose(predictions)[1]
        features_a = np.percentile(probs_class_a, np.arange(0, 101, 5))
        features_b = np.percentile(probs_class_b, np.arange(0, 101, 5))
        return np.concatenate((features_a, features_b), axis=0)

    # internal wrapper that invokes training and
    # evaluation of the meta regressor
    def validate(f):
        def wrapper(*args):
            message = ("WARNING! Performance drop: %.4f > %.2f, "
                       + "scores deviate by %.4f")
            learner = f(*args)
            model = learner.model
            perturbations = learner.perturbations

            # Training
            print("\nTraining meta regressor on perturbed test data.")
            meta_regressor = train_random_forest_regressor(X_test, y_test,
                                                           perturbations,
                                                           model, learner)

            # Evaluation
            print("\nEvaluating meta regressor on perturbed target data.")
            threshold = .01
            features = percentiles_of_probas(model.predict_proba(X_target))
            predicted_score = meta_regressor.predict(features.reshape(1, -1))
            real_score = model.score(X_target, y_target)
            diff = np.abs(real_score - predicted_score)
            ratio = diff / real_score
            print(diff, ratio)
            if ratio > threshold:
                print(message % (ratio, threshold, diff))
            else:
                print("Everything is fine")

            return model
        return wrapper
    return validate


def main():
    """
    """
    pass


if __name__ == "__main__":
    main()
