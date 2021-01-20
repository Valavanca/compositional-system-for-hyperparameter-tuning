from __future__ import annotations

import logging
from abc import ABC, abstractmethod

class Surrogate(ABC):

    def __init__(self):
        """ might accept constants as arguments that determine the surrogate’s behavior
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def fit(self, X, y, **fit_params) -> self:
        """ Fit the surrogate according to the given training data.

        Args:
            X (array-like, sparse matrix): shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

            y (array-like): shape (n_samples, n_objectives)
            Target vector relative to X

        Returns:
            self: An instance of the surrogate.
        """        
        pass

    @abstractmethod
    def predict(self, X, **predict_params):
        pass

    @abstractmethod
    def score(self, X_test, y_test, callback, **callback_params):
        pass
