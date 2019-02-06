import pandas as pd
import numpy as np
import os


class TrollingDataset:

    def __init__(self):
        self.df = pd.read_csv('resources/data/trolls/data.tsv', sep='\t')
        self.categorical_columns = []
        self.numerical_columns = []
        self.textual_columns = ['content']

    def name(self):
        return "trolling_tweets"

    def labels_from(self, dataframe):
        return np.array(dataframe.label == 1)


class BalancedTrollingDataset:

    def __init__(self):
        complete_data = pd.read_csv('resources/data/trolls/data.tsv', sep='\t')

        trolling = complete_data[complete_data.label == 1]
        not_trolling = complete_data[complete_data.label != 1].sample(len(trolling))

        self.categorical_columns = []
        self.numerical_columns = []
        self.textual_columns = ['content']
        self.df = pd.concat([trolling, not_trolling])

    def name(self):
        return "trolling_tweets_balanced"

    def labels_from(self, dataframe):
        return np.array(dataframe.label == 1)


class AdultDataset:

    def __init__(self):
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            '..resources/data/adult/adult.csv')
        self.df = pd.read_csv(path)
        self.categorical_columns = ['workclass', 'occupation', 'marital_status', 'education']
        self.numerical_columns = ['hours_per_week', 'age']
        self.textual_columns = []

    def name(self):
        return "adult_income"

    def labels_from(self, dataframe):
        return np.array(dataframe['class'] == '>50K')


class BalancedAdultDataset:

    def __init__(self):
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            '../../resources/data/adult/adult.csv')
        complete_data = pd.read_csv(path)
        rich = complete_data[complete_data['class'] == '>50K']
        not_rich = complete_data[complete_data['class'] != '>50K'].sample(len(rich))
        self.categorical_columns = ['workclass', 'occupation', 'marital_status', 'education']
        self.numerical_columns = ['hours_per_week', 'age']
        self.textual_columns = []
        self.df = pd.concat([rich, not_rich])

    def name(self):
        return "adult_income_balanced"

    def labels_from(self, dataframe):
        return np.array(dataframe['class'] == '>50K')


class BankmarketingDataset:

    def __init__(self):
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            '../../resources/data/bankmarketing/bank-full.csv')
        self.df = pd.read_csv(path, sep=';')
        self.categorical_columns = ['job', 'marital', 'housing', 'contact', 'default']
        self.numerical_columns = ['balance', 'age']
        self.textual_columns = []

    def name(self):
        return "bank_marketing"

    def labels_from(self, dataframe):
        return np.array(dataframe.y == 'yes')


class BalancedBankmarketingDataset:

    def __init__(self):
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            '../../resources/data/bankmarketing/bank-full.csv')
        complete_data = pd.read_csv(path, sep=';')
        subscribed = complete_data[complete_data.y == 'yes']
        subscribed_not = complete_data[complete_data.y != 'yes'].sample(len(subscribed))

        self.categorical_columns = ['job', 'marital', 'housing', 'contact', 'default']
        self.numerical_columns = ['balance', 'age']
        self.textual_columns = []
        self.df = pd.concat([subscribed, subscribed_not])

    def name(self):
        return "bank_marketing_balanced"

    def labels_from(self, dataframe):
        return np.array(dataframe.y == 'yes')


class CardioDataset:

    def __init__(self):
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            '../../resources/data/cardio/cardio_train.csv')
        data = pd.read_csv(path, sep=';')
        data['bmi'] = data['weight'] / (.01 * data['height']) ** 2
        data['age_in_years'] = data['age'] / 365

        self.categorical_columns = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
        self.numerical_columns = ['age_in_years', 'ap_hi', 'ap_lo', 'bmi']
        self.textual_columns = []
        self.df = data

    def name(self):
        return "cardio_vascular_diseases"

    def labels_from(self, dataframe):
        return np.array(dataframe.cardio == 1)
