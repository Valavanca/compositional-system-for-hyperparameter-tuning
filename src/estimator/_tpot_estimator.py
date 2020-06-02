from typing import Union
import warnings
warnings.filterwarnings('ignore')

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted, check_X_y

from src.composite import ModelsUnion
from tpot import TPOTRegressor, TPOTClassifier

class TpotEstimator(BaseEstimator):
    """ This class is a wrapper for TPOT estimator that allows estimate multi labels date in isolation and provide sklearn compatible methods.

    Args:
        est_type (str, optional): TPOT estimator type. Defaults to "regressor".

    Ref:
        TPOT: http://epistasislab.github.io/tpot/
    """

    def __init__(self, est_type: str = "regressor", **tpot_params):     
        self._type = est_type
        self._tpot_params = tpot_params
        self._dev_pipes = None
        self._union = None

    def fit(self, X, y):
        X, y = check_X_y(X, y, estimator=self, multi_output=True)

        models = []
        if self._type == "regressor":
            print("TPOT regressor")
            for _ in range(y.shape[1]):
                models.append(TPOTRegressor(**self._tpot_params))
        elif self._type == "classifier":
            print("TPOT classifier")
            for _ in range(y.shape[1]):
                models.append(TPOTClassifier(**self._tpot_params))
        else:
            raise Exception(
                "Wrong estimator type: {}. Should be 'regressor' or 'classifier'".format(self._type))
        self._dev_pipes = [
            m.fit(X, y[:, idx]).fitted_pipeline_ for idx, m in enumerate(models, 0)]
        self._union = ModelsUnion(models=self._dev_pipes, split_y=True)
        return self

    def predict(self, X):
        return self._union.predict(X)

    def score(self, **score_params):
        return self._union.score(**score_params)

    def cross_validate(self, X, y, **cv_params):
        self.fit(X, y)
        return self._union.cross_validate(X, y, **cv_params)
