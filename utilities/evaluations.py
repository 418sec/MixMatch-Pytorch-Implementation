import numpy as np
from operator import eq
from sklearn import metrics

def accuracy(y_true, y_pred):
    balanced_accuracy = metrics.accuracy_score(y_true, y_pred)
    return balanced_accuracy