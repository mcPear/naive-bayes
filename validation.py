from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy as np
import utils


def ten_fold_stratified(data, estimator, scoring):
    attrs, classes = utils.horizontal_split(data)
    cv = StratifiedKFold(n_splits=10, shuffle=True)
    scores = cross_val_score(estimator, attrs, classes, cv=cv, scoring=scoring)
    mean_score = np.mean(scores)
    return mean_score
