#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

""""""

from sklearn.model_selection import train_test_split as split
from quilt.data.usr import wine, credit
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt
from scipy.stats import ks_2samp
from copy import deepcopy
import numpy as np

from pipelines import WineQualityMissingPipeline, CreditGPipeline
from pipelines import WineQualityPipeline
from models import RandomForest
from messages import Message


class Histogram:
    def __init__(self, data, n_bins=100, borders=None):
        self.n_bins = n_bins
        self.data_size = data.shape[0]
        if borders is None:
            self.min_val = np.min(data)
            self.max_val = np.max(data)
        else:
            isinstance(borders, tuple), "borders must be a tuple"
            len(borders) == 2, "borders is a tuple of size 2"
            self.min_val, self.max_val = borders
        mean_val = .5 * (self.max_val + self.min_val)
        std_val = self.max_val - mean_val
        bins = np.linspace(-1.001, 1.001, num=self.n_bins, endpoint=True)
        norm_data = (data - mean_val) / std_val
        # print(data)
        # print(norm_data)
        tmp = np.digitize(norm_data, bins)
        self.binned = np.array([float(len(norm_data[tmp == i]))/self.data_size
                                if (tmp == i).any() else .0
                                for i in range(1, len(bins))])
        print(self.binned)
        print

    def __foo(self, hist, bins, value):
        bin_idx = np.argmax(bins > value)
        left, right = bins[bin_idx], bins[bin_idx + 1]
        factor = float(value - left) / (right - left)
        # print(hist.shape, bins.shape)
        return (np.sum(hist[bins <= value]) + hist[bin_idx] * factor)

    def update_with(self, other):
        if other.n_bins == self.n_bins:
            new_histogram = other.binned
        else:
            new_bins = np.linspace(-1., 1., num=self.n_bins, endpoint=True)
            other_hist = other.binned * other.data_size
            new_histogram = [(self.__foo(other_hist, new_bins, right)
                              - self.__foo(other_hist, new_bins, left))
                             for left, right in zip(new_bins, new_bins[1:])]
        factor = 1. / (self.data_size + other.data_size)
        self.binned = factor * (self.binned + new_histogram)


class SklearnDataShiftDetector:
    def __init__(self, pipeline, n_bins=100):
        assert isinstance(pipeline, Pipeline), Message().not_a_pipeline
        self.pipeline = pipeline
        print(self.pipeline)
        self.history = []
        self.n_bins = n_bins

    def on(self, data):
        return self.run(data)

    def iteration(self, data):
        if len(self.history) == 0:
            for i in range(len(self.pipeline.steps)-1):
                pipeline_chunk = Pipeline(self.pipeline.steps[:i+1])
                self.history.append(Histogram(pipeline_chunk.transform(data),
                                              n_bins=self.n_bins))
            self.history.append(Histogram(self.pipeline.predict(data),
                                          n_bins=10))
        else:
            old_hist = self.history
            for i in range(len(self.pipeline.steps)-1):
                pipeline_chunk = Pipeline(self.pipeline.steps[:i+1])
                self.history[i].update_with(
                    Histogram(pipeline_chunk.transform(data),
                              n_bins=self.n_bins))
            self.history[-1].update_with(
                Histogram(self.pipeline.predict(data), n_bins=10))
            results = []
            for prev, cur in zip(old_hist, self.history):
                results.append(ks_2samp(prev.binned, cur.binned))
            self.ks_stats, self.p_values = zip(*results)
        return self

    def data_is_shifted(self, threshold=0.05):
        print(self.p_values)
        return (np.array(self.p_values) < threshold).any()


def main():
    """
    """
    # data = credit.dataset_31_credit_g()
    data = wine.wine_quality_red_csv()
    print(data.columns)
    # column = data['volatile_acidity'].values.reshape(-1, 1)
    # column = data[].values.reshape(-1, 1)
    X, y = data['volatile_acidity'].values.reshape(-1, 1), data['class']
    X_train, X_test, y_train, y_test = split(X, y,
                                             test_size=0.2,
                                             random_state=0)
    sets = split(X_test, y_test, test_size=.5, random_state=0)
    X_first_half, X_second_half, y_first_half, y_second_half = sets
    # print(X_first_half.shape, X_second_half.shape)
    # X_train, X_test, y_train, y_test = split(X, y,
    #                                          test_size=0.2,
    #                                          random_state=0)

    pipeline = WineQualityPipeline()
    classifier = RandomForest()
    model = pipeline.with_estimator(classifier).fit(X_train, y_train)
    # prediction = model.predict(X_test)
    # pipeline = CreditGPipeline()
    shift_detector = SklearnDataShiftDetector(model, n_bins=30)
    shift_detector.iteration(X_first_half)
    new_second_half = deepcopy(X_second_half)
    mask = np.logical_and(X_second_half > .4, X_second_half < 1.)
    new_second_half[mask] *= 3.
    plt.plot(range(X_first_half.shape[0]), X_first_half, 'go')
    plt.plot(range(new_second_half.shape[0]), new_second_half, 'r^')
    plt.show()
    shift_detector.iteration(new_second_half)
    print(shift_detector.data_is_shifted())


if __name__ == "__main__":
    main()
