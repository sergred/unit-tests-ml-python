import random
import numpy as np


class MissingValues:

    def __init__(self, fraction, columns, value_to_put_in):
        self.fraction = fraction
        self.columns = columns
        self.value_to_put_in = value_to_put_in

    def transform(self, clean_df):
        # we operate on a copy of the data
        df = clean_df.copy(deep=True)

        row_indexes = [row for row in range(df.shape[0])]
        # pick random examples
        num_rows_to_pick = int(round(self.fraction * len(row_indexes)))
        for random_row_index in random.sample(row_indexes, num_rows_to_pick):
            # delete all specified values in the target columns
            for column in self.columns:
                column_index = df.columns.get_loc(column)
                df.iat[random_row_index, column_index] = self.value_to_put_in

        return df


class Leetspeak:

    def __init__(self, fraction, column, label_column, label_value):
        self.fraction = fraction
        self.column = column
        self.label_column = label_column
        self.label_value = label_value

    def transform(self, clean_df):
        # we operate on a copy of the data
        df = clean_df.copy(deep=True)

        for index, row in df.iterrows():

            if row[self.label_column] == self.label_value and random.random() < self.fraction:
                leet_content = row[self.column] \
                    .replace('a', '4') \
                    .replace('e', '3') \
                    .replace('l', '1') \
                    .replace('t', '7') \
                    .replace('s', '5') \
                    .replace('o', '0')
                df.at[index, self.column] = leet_content

        return df


class Outliers:

    def __init__(self, fraction, columns):
        self.fraction = fraction
        self.columns = columns

    def transform(self, clean_df):
        # we operate on a copy of the data
        df = clean_df.copy(deep=True)

        means = {column: np.mean(df[column]) for column in self.columns}
        stddevs = {column: np.std(df[column]) for column in self.columns}
        scales = {column: random.uniform(2, 5) for column in self.columns}

        for index, row in df.iterrows():

            for column in self.columns:
                if random.random() < self.fraction:

                    outlier = np.random.normal(means[column], scales[column] * stddevs[column])

                    df.at[index, column] = outlier

        return df
