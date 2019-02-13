import numpy as np
from copy import deepcopy
import random as rand


class MissingValues:

    def __init__(self, value_to_put_in):
        self.fraction = None
        self.columns = None
        self.value_to_put_in = value_to_put_in

    def on(self, fraction, columns):
        tmp = deepcopy(self)
        tmp.fraction = fraction
        tmp.columns = columns
        return tmp

    def transform(self, clean_df):
        assert self.fraction is not None, "Run .on() method first."
        assert self.columns is not None, "Run .on() method first."
        # we operate on a copy of the data
        df = clean_df.copy(deep=True)

        row_indexes = [row for row in range(df.shape[0])]
        # pick random examples
        num_rows_to_pick = int(round(self.fraction * len(row_indexes)))
        for random_row_index in rand.sample(row_indexes, num_rows_to_pick):
            # delete all specified values in the target columns
            for column in self.columns:
                column_index = df.columns.get_loc(column)
                df.iat[random_row_index, column_index] = self.value_to_put_in

        return df


class Leetspeak:

    def __init__(self, label_column, label_value):
        self.fraction = None
        self.columns = None
        self.label_column = label_column
        self.label_value = label_value

    def on(self, fraction, columns):
        tmp = deepcopy(self)
        tmp.fraction = fraction
        tmp.columns = columns
        return tmp

    def transform(self, clean_df):
        assert self.fraction is not None, "Run .on() method first."
        assert self.columns is not None, "Run .on() method first."
        # we operate on a copy of the data
        df = clean_df.copy(deep=True)

        mask = df[self.label_column] == self.label_value
        df.loc[mask, self.columns] = df.apply(lambda row: (row[self.columns]
                                                           .replace('a', '4')
                                                           .replace('e', '3')
                                                           .replace('l', '1')
                                                           .replace('t', '7')
                                                           .replace('s', '5')
                                                           .replace('o', '0'))
                                              if rand.random() < self.fraction
                                              else row[self.columns], axis=1)
        return df


class Outliers:

    def __init__(self):
        self.fraction = None
        self.columns = None

    def on(self, fraction, columns):
        tmp = deepcopy(self)
        tmp.fraction = fraction
        tmp.columns = columns
        return tmp

    def transform(self, clean_df):
        assert self.fraction is not None, "Run .on() method first."
        assert self.columns is not None, "Run .on() method first."
        # we operate on a copy of the data
        df = clean_df.copy(deep=True)

        means = {column: np.mean(df[column]) for column in self.columns}
        stddevs = {column: np.std(df[column]) for column in self.columns}
        scales = {column: rand.uniform(2, 5) for column in self.columns}

        for index, row in df.iterrows():

            for column in self.columns:
                if rand.random() < self.fraction:

                    outlier = np.random.normal(means[column], scales[column] * stddevs[column])

                    df.at[index, column] = outlier

        return df


def main():
    from ssc.hilda.datasets import TrollingDataset
    from ssc.hilda.learners import LogisticRegression
    dataset = TrollingDataset()
    learner = LogisticRegression('accuracy')
    X_train, X_test, X_target = learner.split(dataset.df)
    print(Leetspeak('label', 1).on(.5, 'content').transform(X_test))


if __name__ == "__main__":
    main()
