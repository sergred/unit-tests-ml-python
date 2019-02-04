from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, roc_auc_score

import tensorflow as tf
from tensorflow import keras

import warnings
warnings.filterwarnings('ignore')


class Learner:

    def __init__(self, scoring):
        self.scoring = scoring

    def split(self, data):
        train_data, heldout_data = train_test_split(data, test_size=0.2)
        test_data, target_data = train_test_split(heldout_data, test_size=0.5)

        return train_data, test_data, target_data

    def scoring_name(self):
        return self.scoring

    def score(self, y_true, y_pred):
        if self.scoring == 'accuracy':
            return accuracy_score(y_true, y_pred)

        if self.scoring == 'roc_auc':
            return roc_auc_score(y_true, y_pred)

        raise Exception('unknown scoring {}'.format(self.scoring))


class LogisticRegression(Learner):

    def __init__(self, scoring):
        super(LogisticRegression, self).__init__(scoring)
        self.name = "logistic_regression"

    def fit(self, dataset, train_data):

        y_train = dataset.labels_from(train_data)

        feature_transformation = ColumnTransformer(transformers=[
            ('categorical_features', OneHotEncoder(handle_unknown='ignore'), dataset.categorical_columns),
            ('scaled_numeric', StandardScaler(), dataset.numerical_columns)
        ])

        param_grid = {
            'learner__loss': ['log'],
            'learner__penalty': ['l2', 'l1', 'elasticnet'],
            'learner__alpha': [0.0001, 0.001, 0.01, 0.1]
        }

        pipeline = Pipeline([
            ('features', feature_transformation),
            ('learner', SGDClassifier(max_iter=1000))])

        search = GridSearchCV(pipeline, param_grid, scoring=self.scoring, cv=5, verbose=1, n_jobs=-1)
        model = search.fit(train_data, y_train)

        return model


class DNN(Learner):

    def __init__(self, scoring):
        super(DNN, self).__init__(scoring)
        self.name = "dnn"

    def fit(self, dataset, train_data):

        y_train = dataset.labels_from(train_data)

        feature_transformation = ColumnTransformer(transformers=[
            ('categorical_features', OneHotEncoder(handle_unknown='ignore'), dataset.categorical_columns),
            ('scaled_numeric', StandardScaler(), dataset.numerical_columns)
        ], sparse_threshold=0)

        def create_model(size_1, size_2):
            nn = keras.Sequential([
                keras.layers.Dense(size_1, activation=tf.nn.relu),
                keras.layers.Dense(size_2, activation=tf.nn.relu),
                keras.layers.Dense(2, activation=tf.nn.softmax)
            ])

            nn.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy']) # TODO figure out how to use roc_auc here...
            return nn

        nn_model = keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model)

        pipeline = Pipeline([
            ('features', feature_transformation),
            ('learner', nn_model)])

        param_grid = {
            'learner__epochs': [50],
            'learner__batch_size': [1024],
            'learner__size_1': [4, 8],
            'learner__size_2': [2, 4],
            'learner__verbose': [0]
        }

        model = GridSearchCV(pipeline, param_grid, scoring=self.scoring, cv=5, verbose=1).fit(train_data, y_train)

        # print(model.cv_results_)

        return model
