import numpy as np
import pandas as pd
from copy import deepcopy

from sklearn.multioutput import RegressorChain
from sklearn.utils.validation import check_X_y
from sklearn.multioutput import RegressorChain

from joblib import Parallel, delayed

from src.surrogate.surrogate_abs import Surrogate


class Union(Surrogate):

    def __init__(self,
                 surrogates: [],
                 union_type: str = "average",
                 is_multi_obj: bool = True,
                 n_jobs: int = -1,
                 **union_params
                 ):
        """
        Args:
            surrogates (list): surrogates

            union_type (str, optional): approach that specify how to combine estinations from several surrogates. 
                - "average": train on all dataset and average predictions from each surrogate.
                - "separate": each surrogate trained on all dataset and describe appropriate objective(s). Separate surrogate for each objective variable.
                - "chain": arranges surrogates into a chain. Each surrogate makes a prediction in the order specified by the chain using all of the available features provided to the surrogate plus the predictions of models that are earlier in the chain. 
            Defaults to "average".
        """
        super().__init__()
        self.surrogates = surrogates
        self._surrogates = None
        self.method = "predict"
        self.is_multi_obj = is_multi_obj
        self._union_type = union_type
        self.n_jobs = n_jobs
        self.union_params = union_params

    def fit(self, X, y, sample_weight=None, **fit_params):
        X, y = check_X_y(X, y, multi_output=self.is_multi_obj, accept_sparse=True)
        self._surrogates = None
        self._check_surrogates()
        self._surrogates = deepcopy(self.surrogates)  # work with _surrogates

        if self._union_type == "avarage":  # for single and multi objective
            self._surrogates = self._all_obj_fit(X, y, sample_weight, **fit_params)
        elif self._union_type == "separate":  # for multi objective
            self._check_surr_y(y)
            self._surrogates = self._separate_obj_fit(X, y, sample_weight, **fit_params)
        elif self._union_type == "chain":  # for multi objective
            # dublicate first surrogate and make multi-obj chain prediction
            surr_chain = RegressorChain(self._surrogates[0], **self.union_params)
            surr_chain.fit(X, y, **fit_params)
            self._surrogates = [surr_chain]

        return self

    def _all_obj_fit(self, X, y, sample_weight=None, **fit_params):
        """ Fit a several surrogates on the full dataset.
        """
        fited_surr = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_surrogate)(
                self._surrogates[i], X, y, sample_weight, **fit_params)
            for i in range(len(self._surrogates)))
        return fited_surr

    def _separate_obj_fit(self, X, y, sample_weight=None, **fit_params):
        """ Fit a separate model for each output variable.
        """
        fited_surr = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_surrogate)(
                self._surrogates[i], X, y[:, i], sample_weight, **fit_params)
            for i in range(len(self._surrogates)))
        return fited_surr

    def predict(self, X, **predict_params):
        # 1. prediction from each surrogate
        # prediction = np.array([surr.predict(X) for surr in self._surrogates])

        # prediction = Parallel(n_jobs=self.n_jobs)(
        #     delayed(_predict)(
        #         self._surrogates[i], X, **predict_params)
        #     for i in range(len(self._surrogates)))
        # prediction = np.array(prediction)

        prediction = [_predict(surr, X, **predict_params) for surr in self._surrogates]
        prediction = np.array(prediction)


        # 2. combine predictions
        if self._union_type == "avarage":
            prediction = prediction.mean(axis=0)
        elif self._union_type == "separate":
            prediction = prediction.T
        elif self._union_type == "chain":
            prediction = prediction[0]

        return prediction

    def score(self, X_test, y_test, callback, **callback_params):
        y_pred = self.predict(X_test)
        score = callback(y_test, y_pred, **callback_params)
        return score

    def _check_surrogates(self):
        if self.surrogates is None:
            raise ValueError(
                f"This {self.__class__.__name__} surrogate "
                f"requires surrogate to be passed, but the surrogate set is None."
            )
        else:
            for surr in self.surrogates:
                if not hasattr(surr, self.method):
                    raise ValueError(
                        f"This {surr.__class__.__name__} surrogate "
                        f"doesn't the required method ({self.method})."
                    )

    def _check_surr_y(self, y):
        obj_count = np.size(y, 1)
        surr_count = len(self._surrogates)
        if obj_count != surr_count:
            msg = f"Number of surrogates and the number of objectives must match. It may be necessary to indicate that this is a single-criteria task. \
                Objectives: {obj_count}, Surrogates: {surr_count}."
            raise ValueError(msg)

# -------------------------------------------------------------------
# --- Utilities for parallelization

def _fit_surrogate(surrogate, X, y, sample_weight=None, **fit_params):
    if sample_weight is not None:
        surrogate.fit(X, y, sample_weight=sample_weight, **fit_params)
    else:
        surrogate.fit(X, y, **fit_params)
    return surrogate

def _predict(surrogate, X, **predict_params):
    return surrogate.predict(X, **predict_params)

def _score(surrogate, X_test, y_test, callback, **callback_params):
    y_pred = surrogate.predict(X_test)
    score = callback(y_test, y_pred, **callback_params)
    return score
