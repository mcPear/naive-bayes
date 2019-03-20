from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics.scorer import get_scorer
import numpy as np
from util import utils


def k_fold(data, estimator, scoring, k, stratified=True):
    attrs, classes = utils.horizontal_split(data)
    cv = StratifiedKFold(n_splits=k) if stratified else KFold(n_splits=k)
    scores = cross_validate(estimator, attrs, classes, cv=cv, scoring=scoring)
    scores_test = [scores["test_" + scoring_elem] for scoring_elem in scoring]
    return np.mean(scores_test, (0, 1))  # mean by metrics then by folds


def single_split(data, estimator, scoring):
    attrs, classes = utils.horizontal_split(data)
    X_train, X_test, y_train, y_test = train_test_split(attrs, classes, test_size=0.4)
    estimator.fit(X_train, y_train)
    scorer = get_scorer(scoring)
    return scorer(estimator, X_test, y_test)
