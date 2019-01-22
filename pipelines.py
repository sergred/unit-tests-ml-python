#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

""""""

from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from copy import deepcopy

from transformers import OneHotEncodingTransformer, OrdinalScaleTransformer
from transformers import DenseTransformer
from analyzers import DataType, DataScale


class BasePipeline:
    def __init__(self):
        self.pipe = None

    def __getattr__(self, name):
        return getattr(self.pipe, name)

    def with_estimator(self, estimator):
        complete_pipeline = deepcopy(self.pipe)
        # complete_pipeline.steps.append(('to_dense', DenseTransformer()))
        complete_pipeline.steps.append(('estimator', estimator))
        return complete_pipeline


class AutomatedPipeline(BasePipeline):
    def __init__(self, data_profile):
        self.pipe = None
        self.profile = data_profile.profiles
        self.run()

    def run(self):
        num_ord_features = [col.column_name for col in self.profile
                            if col.dtype in [DataType.INTEGER, DataType.FLOAT]]
        num_cat_features = [col.column_name for col in self.profile
                            if col.dtype in [DataType.INTEGER, DataType.FLOAT]
                            and col.scale == DataScale.NOMINAL]
        str_cat_features = [col.column_name for col in self.profile
                            if col.dtype in [DataType.STRING]
                            and col.scale == DataScale.NOMINAL]
        # for column_profile in self.profile:
        #     print(column_profile)

        num_ord_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean'))])

        num_cat_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        str_cat_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant',
                                      fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        transformers = ColumnTransformer(transformers=[
            ('num_ord', num_ord_transformer, num_ord_features),
            ('num_cat', num_cat_transformer, num_cat_features),
            ('str_cat', str_cat_transformer, str_cat_features)])

        self.pipe = Pipeline([
            ('preprocessing', transformers),
            ('to_dense', DenseTransformer()),
            ('scaler', StandardScaler())])


class CreditGPipeline(BasePipeline):
    def __init__(self):
        """
        checking_status           1000 non-null object
        duration                  1000 non-null int64
        credit_history            1000 non-null object
        purpose                   1000 non-null object
        credit_amount             1000 non-null int64
        savings_status            1000 non-null object
        employment                1000 non-null object
        installment_commitment    1000 non-null int64
        personal_status           1000 non-null object
        other_parties             1000 non-null object
        residence_since           1000 non-null int64
        property_magnitude        1000 non-null object
        age                       1000 non-null int64
        other_payment_plans       1000 non-null object
        housing                   1000 non-null object
        existing_credits          1000 non-null int64
        job                       1000 non-null object
        num_dependents            1000 non-null int64
        own_telephone             1000 non-null object
        foreign_worker            1000 non-null object
        class                     1000 non-null object
        """

        """
        @checking_status: ["'0<=X<200'", "'<0'", "'>=200'", "'no checking'"]
        @credit_history: ["'all paid'", "'critical/other existing credit'",
        "'delayed previously'", "'existing paid'", "'no credits/all paid'"]
        purpose: ["'domestic appliance'", "'new car'", "'used car'",
        'business', 'education', 'furniture/equipment', 'other', 'radio/tv',
        'repairs', 'retraining']
        @savings_status: ["'100<=X<500'", "'500<=X<1000'", "'<100'",
        "'>=1000'", "'no known savings'"]
        @employment: ["'1<=X<4'", "'4<=X<7'", "'<1'", "'>=7'", 'unemployed']
        !installment_commitment: [1, 2, 3, 4]
        personal_status: ["'female div/dep/mar'", "'male div/sep'", "'male
        mar/wid'", "'male single'"]
        other_parties: ["'co applicant'", 'guarantor', 'none']
        !residence_since: [1, 2, 3, 4]
        property_magnitude: ["'life insurance'", "'no known property'",
        "'real estate'", 'car']
        other_payment_plans: ['bank', 'none', 'stores']
        housing: ["'for free'", 'own', 'rent']
        !existing_credits: [1, 2, 3, 4]
        job: ["'high qualif/self emp/mgmt'", "'unemp/unskilled non res'",
        "'unskilled resident'", 'skilled']
        !num_dependents: [1, 2]
        own_telephone: ['none', 'yes']
        foreign_worker: ['no', 'yes']
        !class: ['bad', 'good']
        """
        ordering = {
            "checking_status": ["'no checking'", "'<0'", "'0<=X<200'",
                                "'>=200'"],
            "credit_history": ["'all paid'",
                               "'critical/other existing credit'",
                               "'delayed previously'", "'existing paid'",
                               "'no credits/all paid'"],
            "savings_status": ["'no known savings'", "'<100'", "'100<=X<500'",
                               "'500<=X<1000'", "'>=1000'"],
            "employment": ["unemployed", "'<1'", "'1<=X<4'",
                           "'4<=X<7'", "'>=7'"]
        }
        categorical_features = {
            "purpose": ["'domestic appliance'", "'new car'", "'used car'",
                        'business', 'education', 'furniture/equipment',
                        'other', 'radio/tv', 'repairs', 'retraining'],
            "personal_status": ["'female div/dep/mar'", "'male div/sep'",
                                "'male mar/wid'", "'male single'"],
            "other_parties": ["'co applicant'", 'guarantor', 'none'],
            "property_magnitude": ["'life insurance'", "'no known property'",
                                   "'real estate'", 'car'],
            "other_payment_plans": ['bank', 'none', 'stores'],
            "housing": ["'for free'", 'own', 'rent'],
            "job": ["'high qualif/self emp/mgmt'", "'unemp/unskilled non res'",
                    "'unskilled resident'", 'skilled'],
            "own_telephone": ['none', 'yes'],
            "foreign_worker": ['no', 'yes']
        }

        # def foo(ord):
        #     return np.vectorize(lambda x: ord.index(x))

        # ordinal_scalers = ColumnTransformer(
        #     [("%s_idx" % col, ft(foo(ord), validate=False), col)
        #      for col, ord in ordering.items()], remainder='passthrough')
        # one_hot_encoders = ColumnTransformer(
        #     [("%s_idx" % col,
        #       OneHotEncoder(categories=[val],
        #                     n_values=[len(val)],
        #                     sparse=False), col)
        #      for col, val in categorical_features.items()],
        #     remainder='passthrough')
        ordinal_scalers = ColumnTransformer(
            [("%s_idx" % col, OrdinalScaleTransformer(ord), col)
             for col, ord in ordering.items()]
            # , remainder='passthrough')
            # one_hot_encoders = ColumnTransformer(
            + [("%s_idx" % col, OneHotEncodingTransformer(val), col)
               for col, val in categorical_features.items()],
            remainder='passthrough')
        self.pipe = Pipeline([('column transformers', ordinal_scalers),
                              # ('one hot encoders', one_hot_encoders),
                              ('scaler', StandardScaler())])


class WineQualityPipeline(BasePipeline):
    def __init__(self):
        """
        fixed_acidity           1599 non-null float64
        volatile_acidity        1599 non-null float64
        citric_acid             1599 non-null float64
        residual_sugar          1599 non-null float64
        chlorides               1599 non-null float64
        free_sulfur_dioxide     1599 non-null float64
        total_sulfur_dioxide    1599 non-null float64
        density                 1599 non-null float64
        pH                      1599 non-null float64
        sulphates               1599 non-null float64
        alcohol                 1599 non-null float64
        class                   1599 non-null int64
        """
        self.pipe = Pipeline([('scaler', StandardScaler())])


class WineQualityMissingPipeline(BasePipeline):
    def __init__(self):
        """
        fixed_acidity           1599 non-null float64
        volatile_acidity        1599 non-null float64
        citric_acid             1599 non-null float64
        residual_sugar          1599 non-null float64
        chlorides               1599 non-null float64
        free_sulfur_dioxide     1599 non-null float64
        total_sulfur_dioxide    1599 non-null float64
        density                 1599 non-null float64
        pH                      1599 non-null float64
        sulphates               1599 non-null float64
        alcohol                 1599 non-null float64
        class                   1599 non-null int64
        """
        self.pipe = Pipeline([('imputer', SimpleImputer(strategy="mean")),
                              ('scaler', StandardScaler())])


class AbalonePipeline(BasePipeline):
    def __init__(self):
        """

        """
        one_hot_encoders = ColumnTransformer(
            [("Sex_idx", OneHotEncodingTransformer(['M', 'F', 'I']), "Sex")],
            remainder='passthrough')
        self.pipe = Pipeline([('one hot encoders', one_hot_encoders),
                              ('scaler', StandardScaler())])


class AdultPipeline(BasePipeline):
    def __init__(self):
        """
        @age: [17, 18, 19, ... 87, 88, 90]
        @workclass: [ ?,  Federal-gov,  Local-gov,  Never-worked,  Private,
        Self-emp-inc,  Self-emp-not-inc,  State-gov,  Without-pay]
        @fnlwgt: [12285, 13769, 14878, 18827, ... 1366120, 1455435, 1484705]
        @education: [ 10th,  11th,  12th,  1st-4th,  5th-6th,  7th-8th,  9th,
        Assoc-acdm,  Assoc-voc,  Bachelors,  Doctorate,  HS-grad,  Masters,
        Preschool,  Prof-school,  Some-college]
        @education_num: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        @marital_status: [ Divorced,  Married-AF-spouse,  Married-civ-spouse,
        Married-spouse-absent,  Never-married,  Separated,  Widowed]
        @occupation: [ ?,  Adm-clerical,  Armed-Forces,  Craft-repair,
        Exec-managerial,  Farming-fishing,  Handlers-cleaners,
        Machine-op-inspct,  Other-service,  Priv-house-serv,  Prof-specialty,
        Protective-serv,  Sales,  Tech-support,  Transport-moving]
        @relationship: [ Husband,  Not-in-family,  Other-relative,  Own-child,
        Unmarried,  Wife]
        @race: [ Amer-Indian-Eskimo, Asian-Pac-Islander, Black, Other, White]
        @sex: [ Female,  Male]
        @capital_gain: [0, 114, 401, ... 34095, 41310, 99999]
        @capital_loss: [0, 155, 213, ... 3770, 3900, 4356]
        @hours_per_week: [1, 2, 3, ... 97, 98, 99]
        @native_country: [ ?,  Cambodia,  Canada,  China,  Columbia,  Cuba,
        Dominican-Republic,  Ecuador,  El-Salvador,  England,  France,
        Germany,  Greece,  Guatemala,  Haiti,  Holand-Netherlands,  Honduras,
        Hong,  Hungary,  India,  Iran,  Ireland,  Italy,  Jamaica,  Japan,
        Laos,  Mexico,  Nicaragua,  Outlying-US(Guam-USVI-etc),  Peru,
        Philippines,  Poland,  Portugal,  Puerto-Rico,  Scotland,  South,
        Taiwan,  Thailand,  Trinadad&Tobago,  United-States,  Vietnam,
        Yugoslavia]
        @class: [ <=50K,  >50K]
        """
        ordering = ['Preschool', '1st-4th', '5th-6th', '7th-8th',
                    '9th', '10th', '11th', '12th', 'HS-grad',
                    'Prof-school', 'Some-college', 'Assoc-voc',
                    'Assoc-acdm', 'Bachelors', 'Masters',
                    'Doctorate']
        categorical_features = {
            'workclass': ['?', 'Federal-gov', 'Local-gov', 'Never-worked',
                          'Private', 'Self-emp-inc', 'Self-emp-not-inc',
                          'State-gov', 'Without-pay'],
            'marital_status': ['Divorced', 'Married-AF-spouse',
                               'Married-civ-spouse', 'Married-spouse-absent',
                               'Never-married', 'Separated', 'Widowed'],
            'occupation': ['?', 'Adm-clerical', 'Armed-Forces', 'Craft-repair',
                           'Exec-managerial', 'Farming-fishing',
                           'Handlers-cleaners', 'Machine-op-inspct',
                           'Other-service', 'Priv-house-serv',
                           'Prof-specialty', 'Protective-serv', 'Sales',
                           'Tech-support', 'Transport-moving'],
            'relationship': ['Husband', 'Not-in-family', 'Other-relative',
                             'Own-child', 'Unmarried', 'Wife'],
            'race': ['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black',
                     'Other', 'White'],
            'sex': ['Female', 'Male'],
            'native_country': ['?', 'Cambodia', 'Canada', 'China', 'Columbia',
                               'Cuba', 'Dominican-Republic', 'Ecuador',
                               'El-Salvador', 'England', 'France', 'Germany',
                               'Greece', 'Guatemala', 'Haiti',
                               'Holand-Netherlands', 'Honduras', 'Hong',
                               'Hungary', 'India', 'Iran', 'Ireland', 'Italy',
                               'Jamaica', 'Japan', 'Laos', 'Mexico',
                               'Nicaragua', 'Outlying-US(Guam-USVI-etc)',
                               'Peru', 'Philippines', 'Poland', 'Portugal',
                               'Puerto-Rico', 'Scotland', 'South', 'Taiwan',
                               'Thailand', 'Trinadad&Tobago', 'United-States',
                               'Vietnam', 'Yugoslavia']}
        # encoding = ['capital_gain', 'capital_loss']
        # ordinal_scalers = ColumnTransformer(
        #     [("%s_idx" % col, OrdinalScaleTransformer(ord), col)
        #      for col, ord in ordering.items()]
        #     # , remainder='passthrough')
        #     # one_hot_encoders = ColumnTransformer(
        #     + [("%s_idx" % col, OneHotEncodingTransformer(val), col)
        #        for col, val in categorical_features.items()]
        #     # remainder='passthrough')
        #     # lambda_encoders = ColumnTransformer(
        #     + [("%s_idx" % col, LambdaEncodingTransformer(func), col)
        #        for col, func in encoding.items()], remainder='passthrough')
        # self.pipe = Pipeline([('column transformers', ordinal_scalers),
        #                       # ('one hot encoders', one_hot_encoders),
        #                       # ('lambda encoders', lambda_encoders),
        #                       ('scaler', StandardScaler())])
        self.pipe = Pipeline([
            ('transformers', FeatureUnion([
                ('categorical features', Pipeline([
                    ('selector', FunctionTransformer(
                        lambda x: x[list(categorical_features.keys())],
                        validate=False)),
                    ('one hot encoder', OneHotEncoder())])),
                ('ordinal features', Pipeline([
                    ('selector', FunctionTransformer(
                        lambda x: x[['education']],
                        validate=False)),
                    ('one hot encoder',
                     OneHotEncoder(categories=[ordering]))])),
                ('capital encoder', FunctionTransformer(
                    lambda x: ((x[['capital_loss', 'capital_gain']] > 0)
                               .astype(int)),
                    validate=False))])),
            ('scaler', StandardScaler(with_mean=False))])


class AdultMissingPipeline(BasePipeline):
    def __init__(self):
        """
        @age: [17, 18, 19, ... 87, 88, 90]
        @workclass: [ ?,  Federal-gov,  Local-gov,  Never-worked,  Private,
        Self-emp-inc,  Self-emp-not-inc,  State-gov,  Without-pay]
        @fnlwgt: [12285, 13769, 14878, 18827, ... 1366120, 1455435, 1484705]
        @education: [ 10th,  11th,  12th,  1st-4th,  5th-6th,  7th-8th,  9th,
        Assoc-acdm,  Assoc-voc,  Bachelors,  Doctorate,  HS-grad,  Masters,
        Preschool,  Prof-school,  Some-college]
        @education_num: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        @marital_status: [ Divorced,  Married-AF-spouse,  Married-civ-spouse,
        Married-spouse-absent,  Never-married,  Separated,  Widowed]
        @occupation: [ ?,  Adm-clerical,  Armed-Forces,  Craft-repair,
        Exec-managerial,  Farming-fishing,  Handlers-cleaners,
        Machine-op-inspct,  Other-service,  Priv-house-serv,  Prof-specialty,
        Protective-serv,  Sales,  Tech-support,  Transport-moving]
        @relationship: [ Husband,  Not-in-family,  Other-relative,  Own-child,
        Unmarried,  Wife]
        @race: [ Amer-Indian-Eskimo, Asian-Pac-Islander, Black, Other, White]
        @sex: [ Female,  Male]
        @capital_gain: [0, 114, 401, ... 34095, 41310, 99999]
        @capital_loss: [0, 155, 213, ... 3770, 3900, 4356]
        @hours_per_week: [1, 2, 3, ... 97, 98, 99]
        @native_country: [ ?,  Cambodia,  Canada,  China,  Columbia,  Cuba,
        Dominican-Republic,  Ecuador,  El-Salvador,  England,  France,
        Germany,  Greece,  Guatemala,  Haiti,  Holand-Netherlands,  Honduras,
        Hong,  Hungary,  India,  Iran,  Ireland,  Italy,  Jamaica,  Japan,
        Laos,  Mexico,  Nicaragua,  Outlying-US(Guam-USVI-etc),  Peru,
        Philippines,  Poland,  Portugal,  Puerto-Rico,  Scotland,  South,
        Taiwan,  Thailand,  Trinadad&Tobago,  United-States,  Vietnam,
        Yugoslavia]
        @class: [ <=50K,  >50K]
        """
        ordering = ['Preschool', '1st-4th', '5th-6th', '7th-8th',
                    '9th', '10th', '11th', '12th', 'HS-grad',
                    'Prof-school', 'Some-college', 'Assoc-voc',
                    'Assoc-acdm', 'Bachelors', 'Masters',
                    'Doctorate']

        categorical_features = {
            'workclass': ['Federal-gov', 'Local-gov', 'Never-worked',
                          'Private', 'Self-emp-inc', 'Self-emp-not-inc',
                          'State-gov', 'Without-pay'],
            'marital_status': ['Divorced', 'Married-AF-spouse',
                               'Married-civ-spouse', 'Married-spouse-absent',
                               'Never-married', 'Separated', 'Widowed'],
            'occupation': ['Adm-clerical', 'Armed-Forces', 'Craft-repair',
                           'Exec-managerial', 'Farming-fishing',
                           'Handlers-cleaners', 'Machine-op-inspct',
                           'Other-service', 'Priv-house-serv',
                           'Prof-specialty', 'Protective-serv', 'Sales',
                           'Tech-support', 'Transport-moving'],
            'relationship': ['Husband', 'Not-in-family', 'Other-relative',
                             'Own-child', 'Unmarried', 'Wife'],
            'race': ['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black',
                     'Other', 'White'],
            'sex': ['Female', 'Male'],
            'native_country': ['Cambodia', 'Canada', 'China', 'Columbia',
                               'Cuba', 'Dominican-Republic', 'Ecuador',
                               'El-Salvador', 'England', 'France', 'Germany',
                               'Greece', 'Guatemala', 'Haiti',
                               'Holand-Netherlands', 'Honduras', 'Hong',
                               'Hungary', 'India', 'Iran', 'Ireland', 'Italy',
                               'Jamaica', 'Japan', 'Laos', 'Mexico',
                               'Nicaragua', 'Outlying-US(Guam-USVI-etc)',
                               'Peru', 'Philippines', 'Poland', 'Portugal',
                               'Puerto-Rico', 'Scotland', 'South', 'Taiwan',
                               'Thailand', 'Trinadad&Tobago', 'United-States',
                               'Vietnam', 'Yugoslavia']}
        # encoding = {
        #     'capital_gain': lambda x: x > 0,
        #     'capital_loss': lambda x: x > 0
        # }
        # string_values = ordering + list(categorical_features.keys())
        numeric_values = ['age', 'capital_gain', 'capital_loss',
                          'education_num', 'hours_per_week', 'fnlwgt']
        # imputers = FeatureUnion([
        #     ('string imputer', ColumnTransformer(
        #         [("%s_tmp" % col,
        #           Imputer(values=['?'], strategy='most_frequent'), col)
        #          for col in string_values], remainder='passthrough')),
        #     ('numeric imputer', ColumnTransformer(
        #         [("%s_tmp" % col, Imputer(strategy="mean"), col)
        #          for col in numeric_values], remainder='passthrough'))])
        # ordinal_scalers = ColumnTransformer(
        #     [("%s_idx" % col, OrdinalScaleTransformer(ord), "%s_tmp" % col)
        #      for col, ord in ordering.items()], remainder='passthrough')
        # one_hot_encoders = ColumnTransformer(
        #     [("%s_idx" % col, OneHotEncodingTransformer(val), "%s_tmp" % col)
        #      for col, val in categorical_features.items()],
        #     remainder='passthrough')
        # lambda_encoders = ColumnTransformer(
        #     [("%s_idx" % col, LambdaEncodingTransformer(func), col)
        #      for col, func in encoding.items()], remainder='passthrough')

        self.pipe = Pipeline([
            ('transformers', FeatureUnion([
                ('categorical features', Pipeline([
                    ('selector', FunctionTransformer(
                        lambda x: x[list(categorical_features.keys())],
                        validate=False)),
                    ('string imputer', SimpleImputer(
                        missing_values=['?'], strategy='most_frequent')),
                    ('one hot encoder', OneHotEncoder())])),
                ('ordinal features', Pipeline([
                    ('selector', FunctionTransformer(
                        lambda x: x[['education']],
                        validate=False)),
                    ('string imputer', SimpleImputer(
                        missing_values=['?'], strategy='most_frequent')),
                    ('one hot encoder',
                     OneHotEncoder(categories=[ordering]))])),
                ('numeric data', Pipeline([
                    ('selector', FunctionTransformer(
                        lambda x: x[numeric_values], validate=False)),
                    ('numeric imputer', SimpleImputer())])),
                ('capital encoder', FunctionTransformer(
                    lambda x: ((x[['capital_loss', 'capital_gain']] > 0)
                               .astype(int)),
                    validate=False))])),
            ('scaler', StandardScaler(with_mean=False))])


class HeartPipeline(BasePipeline):
    def __init__(self):
        """

        """
        categorical_features = {
            'pain_type': [1, 2, 3, 4],
            'ecg': [0, 1, 2],
            'slope': [1, 2, 3],
            'vessels': [0, 1, 2, 3],
            'thal': [3, 6, 7]
        }
        one_hot_encoders = ColumnTransformer(
            [("%s_idx" % col, OneHotEncodingTransformer(val), col)
             for col, val in categorical_features.items()],
            remainder='passthrough')
        self.pipe = Pipeline([('one hot encoders', one_hot_encoders),
                              ('scaler', StandardScaler())])


def main():
    """
    """


if __name__ == "__main__":
    main()
