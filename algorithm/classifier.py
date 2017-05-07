# Import modules
import math
import numpy as np

from sklearn.linear_model import LogisticRegression

# Import local modules
import logger
import reducer

"""
We need a Classifier which can identify this  f(sequence of
numbers)->label for each row. Since there are more sequence of numbers than the labels,
it is obvious that many sequences lead to the same label
"""


def scikit_classify(X_train, Y_train, X_test, Y_test):
    clf = LogisticRegression()
    clf.fit(X_train, Y_train)
    scikit_score = clf.score(X_test, Y_test)
    print ('score Scikit learn: ', scikit_score)


def classify(X_train, Y_train, X_test, Y_test):


def sigmoid(X):
    return 1 / (1 + np.exp(-X))
