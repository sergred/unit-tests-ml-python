from abc import abstractmethod
import pandas as pd

class Dataset:

    @abstractmethod
    def as_dataframe(self):
        pass

    @abstractmethod
    def categorical_columns(self):
        pass

    @abstractmethod
    def numerical_columns(self):
        pass    @abstractmethod


class AdultDataset(Dataset):

    def __init__(self):
        complete_data = pd.read_csv('../resources/data/adult/adult.csv')

        rich = complete_data[complete_data['class'] == '>50K']
        not_rich = complete_data[complete_data['class'] != '>50K'].sample(len(rich))

        self.df = pd.concat([rich, not_rich])

    def as_dataframe(self):
        return self.df

    def categorical_columns(self):
        return ['workclass', 'occupation', 'marital_status', 'education']

    def numerical_columns(self):
        return ['hours_per_week', 'age']