from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics.scorer import get_scorer
import numpy as np
import utils


def ten_fold(data, estimator, scoring, stratified=True):
    attrs, classes = utils.horizontal_split(data)
    k = 10
    cv = StratifiedKFold(n_splits=k, shuffle=True) if stratified else KFold(n_splits=k, shuffle=True)
    scores = cross_val_score(estimator, attrs, classes, cv=cv, scoring=scoring)
    mean_score = np.mean(scores)
    return mean_score


def single_split(data, estimator, scoring):
    attrs, classes = utils.horizontal_split(data)
    X_train, X_test, y_train, y_test = train_test_split(attrs, classes, test_size=0.4)
    estimator.fit(X_train, y_train)
    scorer = get_scorer(scoring)
    return scorer(estimator, X_test, y_test)
