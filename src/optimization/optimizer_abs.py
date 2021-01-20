from __future__ import annotations

import logging
from abc import ABC, abstractmethod

class Optimizer(ABC):
    """ Optimezer schould find optimal points from the provided surrogates
        Require:
            - surrogate
            - parameters bounds
    """

    def __init__(self, method: str):
        """ might accept constants as arguments that determine the optimizer’s behavior
        """
        super().__init__()
        self._logger = logging.getLogger(__name__)
        self._method = method

    @abstractmethod
    def minimize(self, surrogate, bounds, **min_params) -> []:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass
