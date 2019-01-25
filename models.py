#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

"""Models configuration."""

from visualization_utils import visualize
# from metrics import precision, recall

import numpy as np
import pickle

np.random.seed(0)


class Model:
    def __init__(self):
        self.clf = None
        self.dump = None
        self.name = 'model'

    def __str__(self):
        return str(pickle.loads(self.dump))

    def __repr__(self):
        return str(pickle.loads(self.dump))

    def fit(self, X_train, y_train):
        """"""
        self.clf = pickle.loads(self.dump)
        self.clf.fit(X_train, y_train)

    def predict(self, X_test):
        """"""
        return self.clf.predict(X_test)

    def predict_proba(self, X_test):
        """"""
        return self.clf.predict_proba(X_test)

    def execute(self, X_train, y_train, X_test, y_test, class_names):
        """
        Performs model training and visualizes testing results.

        Keyword arguments:
        clf -- classifier object
        name -- classifier title
        X_train, y_train -- training set
        X_test, y_test -- test set
        """
        print(self.name)
        # scores = cross_val_score(clf, X, labels, cv=20, scoring='accuracy')
        # print Message().accuracy % (scores.mean(), scores.std() * 2)
        # scores = cross_val_score(clf, X, labels, cv=20, scoring='precision')
        # print Message().precision % (scores.mean(), scores.std() * 2)
        # scores = cross_val_score(clf, X, labels, cv=20, scoring='recall')
        # print Message().recall % (scores.mean(), scores.std() * 2)
        self.fit(X_train, y_train)
        predicted = self.predict(X_test)
        visualize(y_test, predicted, class_names)

    def save(self):
        from sklearn.externals import joblib
        joblib.dump(self.clf, self.save_path)


class SVM(Model):
    def __init__(self, kernel='poly'):
        from sklearn import svm
        self.dump = pickle.dumps(
            svm.SVC(kernel=kernel, decision_function_shape='ovr',
                    max_iter=2400, probability=True))
        self.name = 'svm'
        self.save_path = "models/%s.pkl" % (self.name, )


class LinearSVM(Model):
    def __init__(self):
        from sklearn import svm
        self.dump = pickle.dumps(
            svm.LinearSVC(max_iter=2400, probability=True))
        self.name = "lsvm"
        self.save_path = "models/%s.pkl" % (self.name, )


class KNN(Model):
    def __init__(self, n_neighbors):
        from sklearn.neighbors import KNeighborsClassifier
        """KNN with distance-based weight points"""
        self.dump = pickle.dumps(
            KNeighborsClassifier(n_neighbors, weights='distance'))
        self.name = "knn"
        self.save_path = "models/%s.pkl" % (self.name, )


class LinearRegression(Model):
    def __init__(self):
        from sklearn.linear_model import LinearRegression
        self.dump = pickle.dumps(LinearRegression())
        self.name = "linreg"
        self.save_path = "models/%s.pkl" % (self.name, )


class LogRegression(Model):
    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        self.dump = pickle.dumps(LogisticRegression(solver='lbfgs',
                                                    multi_class='auto',
                                                    max_iter=800))
        self.name = "logreg"
        self.save_path = "models/%s.pkl" % (self.name, )


class GausNB(Model):
    def __init__(self):
        from sklearn.naive_bayes import GaussianNB
        self.dump = pickle.dumps(GaussianNB())
        self.name = "gpc"
        self.save_path = "models/%s.pkl" % (self.name, )


class DecisionTree(Model):
    def __init__(self):
        from sklearn.tree import DecisionTreeClassifier
        self.dump = pickle.dumps(
            DecisionTreeClassifier(max_depth=None, min_samples_split=2,
                                   random_state=0))
        self.name = "dtree"
        self.save_path = "models/%s.pkl" % (self.name, )


class RandomForest(Model):
    def __init__(self, size=40):
        from sklearn.ensemble import RandomForestClassifier
        self.dump = pickle.dumps(RandomForestClassifier(n_estimators=size))
        self.name = "rfc"
        self.save_path = "models/%s_%d.pkl" % (self.name, size)


class ExtremelyRandomizedTrees(Model):
    def __init__(self, size):
        from sklearn.ensemble import ExtraTreesClassifier
        self.dump = pickle.dumps(
            ExtraTreesClassifier(n_estimators=size, max_depth=None,
                                 min_samples_split=2, random_state=0))
        self.name = "ertc"
        self.save_path = "models/%s_%d.pkl" % (self.name, size)


class BaggingRandomForest(Model):
    def __init__(self, size):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.ensemble import BaggingClassifier
        self.dump = pickle.dumps(
            BaggingClassifier(RandomForestClassifier(n_estimators=size),
                              max_samples=0.5, max_features=0.5))
        self.name = "brfc"
        self.save_path = "models/%s_%d.pkl" % (self.name, size)


class MLPC(Model):
    def __init__(self, input_size):
        from sklearn.neural_network import MLPClassifier
        self.input_size = input_size
        self.dump = pickle.dumps(MLPClassifier(solver='adam', alpha=1e-4,
                                               learning_rate_init=1e-5,
                                               hidden_layer_sizes=input_size,
                                               random_state=1,
                                               max_iter=4000))
        self.name = "mlpc"
        self.save_path = ("models/%s_%s.pkl" %
                          (self.name, "_".join([str(i) for i in input_size])))


class XGB(Model):
    def __init__(self):
        from xgboost import XGBClassifier
        self.dump = pickle.dumps(XGBClassifier(objctive='multi:softmax'))
        self.name = "xgb"
        self.save_path = "models/%s.pkl" % (self.name, )


def main():
    pass


if __name__ == '__main__':
    main()
