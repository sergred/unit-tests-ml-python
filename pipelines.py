#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

""""""

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from copy import deepcopy
import numpy as np


class BasePipeline:
    def __init__(self):
        self.pipe = None

    def __getattr__(self, name):
        return getattr(self.pipe, name)

    def with_estimator(self, estimator):
        complete_pipeline = deepcopy(self.pipe)
        complete_pipeline.steps.append(['estimator', estimator])
        return complete_pipeline


class OrdinalScaleTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, array):
        self.array = array

    def transform(self, X, *_):
        vfunc = np.vectorize(lambda x: self.array.index(x))
        return vfunc(X).reshape(-1, 1)

    def fit(self, *_):
        return self


class OneHotEncodingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, values):
        self.encoder = OneHotEncoder(categories=values, sparse=False)

    def transform(self, X, *_):
        enc = self.encoder.fit(X.values.reshape(-1, 1))
        return self.encoder.transform(X.values.reshape(-1, 1))

    def fit(self, *_):
        return self


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
        @credit_history: ["'all paid'", "'critical/other existing credit'", "'delayed previously'", "'existing paid'", "'no credits/all paid'"]
        purpose: ["'domestic appliance'", "'new car'", "'used car'", 'business', 'education', 'furniture/equipment', 'other', 'radio/tv', 'repairs', 'retraining']
        @savings_status: ["'100<=X<500'", "'500<=X<1000'", "'<100'", "'>=1000'", "'no known savings'"]
        @employment: ["'1<=X<4'", "'4<=X<7'", "'<1'", "'>=7'", 'unemployed']
        !installment_commitment: [1, 2, 3, 4]
        personal_status: ["'female div/dep/mar'", "'male div/sep'", "'male mar/wid'", "'male single'"]
        other_parties: ["'co applicant'", 'guarantor', 'none']
        !residence_since: [1, 2, 3, 4]
        property_magnitude: ["'life insurance'", "'no known property'", "'real estate'", 'car']
        other_payment_plans: ['bank', 'none', 'stores']
        housing: ["'for free'", 'own', 'rent']
        !existing_credits: [1, 2, 3, 4]
        job: ["'high qualif/self emp/mgmt'", "'unemp/unskilled non res'", "'unskilled resident'", 'skilled']
        !num_dependents: [1, 2]
        own_telephone: ['none', 'yes']
        foreign_worker: ['no', 'yes']
        !class: ['bad', 'good']
        """
        ordering = {
            "checking_status": ["'no checking'", "'<0'", "'0<=X<200'", "'>=200'"],
            "credit_history": ["'all paid'", "'critical/other existing credit'",
                               "'delayed previously'", "'existing paid'", "'no credits/all paid'"],
            "savings_status": ["'no known savings'", "'<100'", "'100<=X<500'", "'500<=X<1000'", "'>=1000'"],
            "employment": ["unemployed", "'<1'", "'1<=X<4'", "'4<=X<7'", "'>=7'"]
        }
        categorical_features = {
            "purpose": ["'domestic appliance'", "'new car'", "'used car'", 'business', 'education',
                        'furniture/equipment', 'other', 'radio/tv', 'repairs', 'retraining'],
            "personal_status": ["'female div/dep/mar'", "'male div/sep'", "'male mar/wid'", "'male single'"],
            "other_parties": ["'co applicant'", 'guarantor', 'none'],
            "property_magnitude": ["'life insurance'", "'no known property'", "'real estate'", 'car'],
            "other_payment_plans": ['bank', 'none', 'stores'],
            "housing": ["'for free'", 'own', 'rent'],
            "job": ["'high qualif/self emp/mgmt'", "'unemp/unskilled non res'", "'unskilled resident'", 'skilled'],
            "own_telephone": ['none', 'yes'],
            "foreign_worker": ['no', 'yes']
        }

        column_transformers = ColumnTransformer(
            [("%s_idx" % col, OrdinalScaleTransformer(ord), col) for col, ord in ordering.items()] +
            [("%s_idx" % col, OneHotEncodingTransformer(val), col) for col, val in categorical_features.items()],
            remainder='passthrough')
        self.pipe = Pipeline([('column transformers', column_transformers)])


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
        column_transformers = ColumnTransformer(
            [("Sex_idx", OneHotEncodingTransformer(['M', 'F', 'I']), "Sex")],
            remainder='passthrough')
        self.pipe = Pipeline([('column transformers', column_transformers),
                              ('scaler', StandardScaler())])


def main():
    """
    """
    pass


if __name__ == "__main__":
    main()
