from typing import List
import hashlib
import logging

import numpy as np
from sklearn import clone
from sklearn.model_selection import cross_validate
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_X_y

from joblib import Parallel, delayed


logger = logging.getLogger()
logger.setLevel(logging.INFO)



class ModelsUnion(TransformerMixin, BaseEstimator):
    """ This class allows parallel fitting of several heterogeneous estimators on
        the same feature space(X) separately in objective space(y). Each estimator
        could extrapolate a specific objective in isolation. Final multi-objective
        prediction gathers from all estimators.

    Args:
        TransformerMixin: Mixin class for all transformers in scikit-learn.
        BaseEstimator: Base class for all estimators in scikit-learn.

    Raises:
        Exception: In the case of splitting objective space, a strict count
        of models is not equal to the dimensionality of objective space.
        Strict means a count of models more than one. In case if provided
        one model it is duplicate to the required number of objectives.

    Returns:
        self: object
    """

    def __init__(self, models: List[BaseEstimator], n_jobs: int = -1, split_y: bool = False):
        """
        Args:
            models (List[BaseEstimator]): 
                Heterogeneous estimators
            n_jobs (int, optional): Defaults to -1.
                The number of jobs to run in parallel for both `fit` and `predict`. 
                Context. `-1 means using all processors. See `joblib.parallel_backend 
                ` <n_jobs>` for more details. 
            split_y (bool, optional): Defaults to False.
                Whether estimators should evaluate objective space(y) partially 
                for each objective in isolation. If False, estimators evaluate on the 
                same dataset. 
        """        
        self._models = models
        self.n_jobs = n_jobs
        self.split_y = split_y

    @property
    def models(self):
        return self._models

    @models.setter
    def models(self, models: List[BaseEstimator]):
        self._models = models

    def fit(self, X, y=None, **fit_params):
        self._check_models_count(y)
        self._models = self._parallel_func(X, y, fit_params, _fit)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, *fit_params)
        return self._models

    def transform(self, X, *arg, **kwargs):
        return X

    def predict(self, X, **predict_params):
        """ Predict using underlying estimators.

        Args:
            X (array_like or sparse matrix, shape (n_samples, n_features)): Samples.

        Returns:
            array, shape (n_samples, n_features): Returns predicted values.
        """        

        pred_result = self._parallel_func(X, None, predict_params, _predict)

        if len(self._models) == 1:
            pred_result = pred_result[0]
        elif self.split_y:
            pred_result = np.vstack(pred_result).T.tolist()
        else:
            pred_result = np.array(pred_result).mean(axis=0).tolist()

        return np.array(pred_result)

    def cross_validate(self, X, y, **cv_params):
        self._check_models_count(y)
        return self._parallel_func(X, y, cv_params, _cv)

    def score(self, X, y=None, **score_params):
        score_result = self._parallel_func(X, y, score_params, _score)
        if len(self._models) == 1:
            score_result = score_result[0]
        else:
            score_result = np.array(score_result).mean()

        return score_result

    def _check_models_count(self, y):
        """ Check the count of available models to extrapolate the required objectives. 
            It is necessary for the case if objectives extrapolate in isolation. 

        Args:
            array-like of shape (n_samples,) or (n_samples, n_objectives): Samples objectives.
        """
        if self._models is None:
            Exception("There are no models")

        if self.split_y:
            if len(self._models) == 1:
                logging.info("Expand the model count for each label. Clone: {}".format(
                    type(self._models[0]).__name__))
                etalon = self._models[0]
                self._models = [clone(etalon) for _ in range(y.shape[1])]
            elif y.shape[1] == len(self._models):
                pass
            else:
                raise Exception(
                    "Count of models is not equal to the dimensionality of objective space.\n Models: {}\
                    You can reduce the number of models to one or equalize the number \
                    of models to the count of objective space dimensions. \
                    [{}] dimension in y-space vs [{}] models length".format(self._models, len(y), len(self._models)))

    def _parallel_func(self, X, y, func_params, func):
        """ Runs function in parallel for each estimator. 
            The index of the estimator corresponds to the index in objective order.
        """
        return Parallel(n_jobs=self.n_jobs)(delayed(func)(
            model, X, y, 
            split_y=self.split_y,
            message_clsname='_ModelsUnion',
            idx=idx,
            message='Parallel contex. idx:{}, length:{}'.format(
                idx, len(self._models)),
            **func_params) for idx, model in enumerate(self._models, 0))


def _cv(estimator, X, y, split_y=False, message_clsname='', idx=0, message=None, **cv_params):
    """ Cross-validation the single estimator.

    Args:
        estimator (BaseEstimator): Singl estimator for cross validation
        X (array_like or sparse matrix, shape (n_samples, n_features)): samples
        y (array, shape (n_samples, n_features)): objective values
        split_y (bool, optional): Spliting objective space. Defaults to False.
        message_clsname (str, optional): Log message. Defaults to ''.
        idx (int, optional): 
            Model positional index. In the case of y-space allocation,
            it corresponds to the evaluation on appropriate objective index. Defaults to 0.
        message (str, optional): Log message. Defaults to None.

    Returns:
        Dict: Cross-validation results
    """
    X, y = check_X_y(X, y, estimator=estimator, multi_output=True)
    if split_y:
        y = y[:, idx]
    X, y = check_X_y(X, y, multi_output=True, estimator=estimator)
    y_index = idx if split_y else "all"

    if hasattr(estimator, 'cross_validate'):
        cv_result = estimator.cross_validate(X, y, **cv_params)
    else:
        cv_result = cross_validate(estimator, X, y, **cv_params)
        cv_result['y_index'] = y_index
        cv_result['model name'] = type(estimator).__name__.lower()
        cv_result['params hash'] = hashlib.md5(
            str(estimator.get_params()).encode('utf-8')).hexdigest()[:4]

        ident = cv_result['model name']+cv_result['params hash']
        cv_result['id'] = hashlib.md5(
            ident.encode('utf-8')).hexdigest()[:6]

    key = "{}_{}".format(y_index, type(estimator).__name__.lower())
    return {key: cv_result}


def _fit(transformer, X, y, split_y=False, message_clsname='', idx=0, message=None, **fit_params):
    """ Fitting the single estimator

    Args:
        estimator (BaseEstimator): Singl estimator for fitting.
        X (array_like or sparse matrix, shape (n_samples, n_features)): samples
        y (array, shape (n_samples, n_features)): objective values
        split_y (bool, optional): Spliting objective space. Defaults to False.
        message_clsname (str, optional): Log message. Defaults to ''.
        idx (int, optional): Model positional index. In the case of y-space allocation,
            it corresponds to the evaluation on appropriate objective index. Defaults to 0.
        message (str, optional): Log message. Defaults to None.

    Returns:
        List or Number: Fit result
    """
    X, y = check_X_y(X, y, estimator=transformer, multi_output=True)
    if split_y:
        y = y[:, idx]
    X, y = check_X_y(X, y, multi_output=True, estimator=transformer)
    return transformer.fit(X, y, **fit_params)


def _score(estimator, X, y, split_y=False, message_clsname='', idx=0, message=None, **score_params):
    """ Score evaluation on the single estimator

    Args:
        estimator (BaseEstimator): Singl estimator for score evaluation.
        X (array_like or sparse matrix, shape (n_samples, n_features)): samples
        y (array, shape (n_samples, n_features)): objective values
        split_y (bool, optional): Spliting objective space. Defaults to False.
        message_clsname (str, optional): Log message. Defaults to ''.
        idx (int, optional): Model positional index. In the case of y-space allocation,
            it corresponds to the evaluation on appropriate objective index. Defaults to 0.
        message (str, optional): Log message. Defaults to None.

    Returns:
        List or Number: Score result
    """
    X, y = check_X_y(X, y, estimator=estimator, multi_output=True)
    if split_y:
        y = y[:, idx]
    X, y = check_X_y(X, y, multi_output=True, estimator=estimator)
    return estimator.score(X, y, **score_params)


def _predict(estimator, X, y=None, split_y=False, message_clsname='', idx=None, message=None, **predict_params):
    """ Prediction from the single estimator

    Args:
        estimator (BaseEstimator): Singl estimator for prediction.
        X (array_like or sparse matrix, shape (n_samples, n_features)): samples
        y (array, shape (n_samples, n_features)): objective values
        split_y (bool, optional): Spliting objective space. Defaults to False.
        message_clsname (str, optional): Log message. Defaults to ''.
        idx (int, optional): Model positional index. In the case of y-space allocation,
            it corresponds to the evaluation on appropriate objective index. Defaults to 0.
        message (str, optional): Log message. Defaults to None.

    Returns:
        List or Number: Prediction
    """
    logging.info(message_clsname, message)
    return estimator.predict(X, **predict_params).tolist()
