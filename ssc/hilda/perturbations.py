import random


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
