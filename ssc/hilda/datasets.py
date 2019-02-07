import pandas as pd
import numpy as np
import os


class AdultDataset:

    def __init__(self):
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            '../../resources/data/adult/adult.csv')
        self.df = pd.read_csv(self.path)
        self.categorical_columns = ['workclass', 'occupation', 'marital_status', 'education']
        self.numerical_columns = ['hours_per_week', 'age']

    def name(self):
        return "adult_income"

    def labels_from(self, dataframe):
        return np.array(dataframe['class'] == '>50K')


class BalancedAdultDataset:

    def __init__(self):
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            '../../resources/data/adult/adult.csv')
        complete_data = pd.read_csv(self.path)
        rich = complete_data[complete_data['class'] == '>50K']
        not_rich = complete_data[complete_data['class'] != '>50K'].sample(len(rich))
        self.categorical_columns = ['workclass', 'occupation', 'marital_status', 'education']
        self.numerical_columns = ['hours_per_week', 'age']
        self.df = pd.concat([rich, not_rich])

    def name(self):
        return "adult_income_balanced"

    def labels_from(self, dataframe):
        return np.array(dataframe['class'] == '>50K')


class BankmarketingDataset:

    def __init__(self):
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            '../../resources/data/bankmarketing/bank-full.csv')
        self.df = pd.read_csv(self.path, sep=';')
        self.categorical_columns = ['job', 'marital', 'housing', 'contact', 'default']
        self.numerical_columns = ['balance', 'age']

    def name(self):
        return "bank_marketing"

    def labels_from(self, dataframe):
        return np.array(dataframe.y == 'yes')


class BalancedBankmarketingDataset:

    def __init__(self):
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            '../../resources/data/bankmarketing/bank-full.csv')
        complete_data = pd.read_csv(self.path, sep=';')
        subscribed = complete_data[complete_data.y == 'yes']
        subscribed_not = complete_data[complete_data.y != 'yes'].sample(len(subscribed))

        self.categorical_columns = ['job', 'marital', 'housing', 'contact', 'default']
        self.numerical_columns = ['balance', 'age']
        self.df = pd.concat([subscribed, subscribed_not])

    def name(self):
        return "bank_marketing_balanced"

    def labels_from(self, dataframe):
        return np.array(dataframe.y == 'yes')


class CardioDataset:

    def __init__(self):
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            '../../resources/data/cardio/cardio_train.csv')
        data = pd.read_csv(self.path, sep=';')
        data['bmi'] = data['weight'] / (.01 * data['height']) ** 2
        data['age_in_years'] = data['age'] / 365

        self.categorical_columns = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
        self.numerical_columns = ['age_in_years', 'ap_hi', 'ap_lo', 'bmi']
        self.df = data

    def name(self):
        return "cardio_vascular_diseases"

    def labels_from(self, dataframe):
        return np.array(dataframe.cardio == 1)
