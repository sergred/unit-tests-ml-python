#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

"""Visualization utilities."""

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
from scipy import interp
import numpy as np
import itertools


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    URL
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    Accessed -- 2017/05/04

    Keyword arguments:
    cm -- data confusion matrix
    classes -- class titles
    normalize -- trigger to enable data normalization
    title -- plot title
    cmap -- color map
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        thresh = 0.5
        print("Normalized confusion matrix")
    else:
        thresh = 0.5 * cm.max()
        print("Confusion matrix, without normalization")

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # print cm

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        # print cm[i, j], thresh
        plt.text(j, i, "%.2f" % (cm[i, j], ),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    """Customized plot alignment"""
    # plt.subplots_adjust(left=0.0, right=0.9, top=0.96, bottom=0.14)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_roc_curve(y_test_set, predicted_probs_set, model_name=""):
    """
    ROC_AUC curve

    URL: http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py

    Keyword arguments:
    y_test_set -- set of test data points
    predicted_probs_set -- set of predicted probabilities per each fold
    """
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    for proba, y_test in zip(predicted_probs_set, y_test_set):
        # print proba, y_test
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_test, proba)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)
    print("Average AUC-ROC : %0.4f (+/- %0.4f)" % (mean_auc, std_auc))
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.06)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(model_name + '. ROC curve')
    plt.legend(loc="lower right")
    plt.show()
    return "ROC score: %.4f (+/- %.4f)" % (mean_auc, 2*std_auc)


def visualize(y_test, predicted, class_names, model_name=""):
    """
    Computes performance metrics (accuracy, precision, recall and f1 scores) and
    plots confusion matrix for the given data.

    Keyword arguments:
    y_test -- test set
    predicted -- predicted data
    """
    print("Accuracy : %.4f" % (accuracy_score(y_test, predicted),))
    print("Precision: %.4f" % (precision_score(y_test, predicted, average='weighted'),))
    print("Recall   : %.4f" % (recall_score(y_test, predicted, average='weighted'),))
    print("F-1 score: %.4f" % (f1_score(y_test, predicted, average='weighted'),))
    print("ROC score: %.4f" % (roc_auc_score(y_test, predicted, average=None),))
    cnf_matrix = confusion_matrix(y_test, predicted)

    """Plot normalized confusion matrix"""
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title = model_name + '. Normalized confusion matrix')
    plt.show()


def main():
    """
    """
    pass


if __name__ == "__main__":
    main()
