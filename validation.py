from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy as np


def ten_fold_stratified(attrs, classes, estimator, scoring):
    cv = StratifiedKFold(n_splits=10, shuffle=True)
    scores = cross_val_score(estimator, attrs, classes, cv=cv, scoring=scoring)
    mean_score = np.mean(scores)
    return mean_score
