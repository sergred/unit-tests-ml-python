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
from messages import Message


class SklearnDataShiftDetector:
    def __init__(self, pipeline):
        assert isinstance(pipeline, Pipeline), Message().not_a_pipeline
        self.pipeline = pipeline
        print(self.pipeline)
        self.history = []
        self.ks_stats, self.p_values = [], []

    def on(self, data):
        return self.run(data)

    def iteration(self, data):
        intermediate_results = []
        for i in range(len(self.pipeline.steps)):
            pipeline_chunk = Pipeline(self.pipeline.steps[:i+1])
            tmp = pipeline_chunk.fit_transform(data)
            # print(tmp)
            intermediate_results.append(tmp)
        self.history.append(intermediate_results)

    def run(self):
        assert len(self.history) > 1, Message().not_enough_history
        for previous, current in zip(self.history, self.history[1:]):
            ks_stats, p_values = [], []
            for prev, cur in zip(previous, current):
                ks_stat, p_value = self.test(prev, cur)
                ks_stats.append(ks_stat)
                p_values.append(p_value)
            self.ks_stats.append(ks_stats)
            self.p_values.append(p_values)
        return self

    def test(self, previous, current, num_bins=1000):
        min_val = np.min([np.min(previous), np.min(current)])
        max_val = np.max([np.max(previous), np.max(current)])
        mean_val = .5 * (max_val + min_val)
        std_val = max_val - mean_val
        bins = np.linspace(min_val, max_val, num_bins)
        norm_previous = (previous - mean_val) / std_val
        norm_current = (current - mean_val) / std_val
        prev = np.digitize(norm_previous, bins)
        cur = np.digitize(norm_current, bins)
        prev_binned = [norm_previous[prev == i].mean()
                       if (prev == i).any() else .0
                       for i in range(1, len(bins))]
        cur_binned = [norm_current[cur == i].mean()
                      if (cur == i).any() else .0
                      for i in range(1, len(bins))]
        return ks_2samp(prev_binned, cur_binned)

    def is_shifted(self, threshold=0.01):
        return (np.array(self.p_values) < threshold).any()


def main():
    """
    """
    # data = credit.dataset_31_credit_g()
    data = wine.wine_quality_red_csv()
    column = data['volatile_acidity'].values.reshape(-1, 1)
    # column = data[].values.reshape(-1, 1)
    first_half, second_half = split(column, test_size=.5, random_state=0)
    print(first_half.shape, second_half.shape)
    # X_train, X_test, y_train, y_test = split(X, y,
    #                                          test_size=0.2,
    #                                          random_state=0)

    pipeline = WineQualityMissingPipeline()
    # pipeline = CreditGPipeline()
    shift_detector = SklearnDataShiftDetector(pipeline.pipe)
    shift_detector.iteration(first_half)
    new_second_half = deepcopy(second_half)
    new_second_half[np.logical_and(second_half > .4, second_half < 1.)] += .6
    plt.plot(range(first_half.shape[0]), first_half, 'go')
    plt.plot(range(new_second_half.shape[0]), new_second_half, 'r^')
    # plt.show()
    shift_detector.iteration(new_second_half)
    res = shift_detector.run()
    print(res.p_values)
    print(res.is_shifted())


if __name__ == "__main__":
    main()
