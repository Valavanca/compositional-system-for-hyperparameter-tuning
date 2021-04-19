from abc import abstractmethod

from sklearn.base import BaseEstimator, MetaEstimatorMixin

class Solver(BaseEstimator, MetaEstimatorMixin):
    def __init__(self, estimator):
        self._estimator = estimator

    @property
    def estimator(self):
        return self._estimator

    @abstractmethod
    def fit(self, X, y=None, **fit_params):
        pass

    @abstractmethod
    def predict(self, X=None, **predict_params):
        pass

    @abstractmethod
    def get_name(self):
        pass
