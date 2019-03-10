from sklearn.base import BaseEstimator


class DummyClassifier(BaseEstimator):

    def fit(self, X, y):
        self.y = y
        return self

    def predict(self, X):
        distinct = list(set(self.y))
        size = len(X)
        return (distinct * size)[:size]
