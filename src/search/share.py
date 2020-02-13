from typing import List, Tuple

import pygmo as pg
import numpy as np
from sklearn.base import BaseEstimator


def make_nd_pop(pro, x, y):
    nd_front = pg.fast_non_dominated_sorting(y)[0][0]
    nd_x = x[nd_front]
    nd_y = y[nd_front]
    t_pop = pg.population(pro)
    for i, p_vector in enumerate(nd_x):
        t_pop.push_back(x=p_vector, f=nd_y[i])
    return t_pop

class Pagmo_problem:
    def __init__(self, models: List[BaseEstimator], bounds: Tuple[List] = None, is_single=False, model_weight=None):
        self._estimators = models
        self.__target_func = 'predict'
        self._bounds = bounds
        self._is_single = is_single
        self._weight = model_weight or [1]*len(models)

    def fitness(self, x):
        return self._singl_obj(x) if self._is_single else self._multi_obj(x)

    def _multi_obj(self, x):
        f_vector = [self.__evaluate(e, x) for e in self._estimators]
        return np.multiply(f_vector, self._weight).flatten().tolist()

    def _singl_obj(self, x):
        fw_vector = self._multi_obj(x)
        return [np.mean(fw_vector)]

    def __evaluate(self, estimator, x):
        # result = getattr(estimator, self.__target_func)(x.reshape(-1, 1))
        result = getattr(estimator, self.__target_func)(x.reshape(1, -1))
        return result.tolist()[0]

    def get_nobj(self):
        nobj = None
        if self._is_single:
            nobj = 1
        else:
            if len(self._estimators) > 1:
                nobj = len(self._estimators)
            else:
                t_point = [d[0] for d in self._bounds]
                prediction = self._estimators[0].predict([t_point])
                nobj = len(prediction[0])

        return nobj

    # Return bounds of decision variables
    def get_bounds(self):
        return self._bounds

    def set_bounds(self, bounds: Tuple[List]):
        self._bounds = bounds
        return self

    # Return function name
    def get_name(self):
        return " vs ".join([type(t).__name__ for t in self._estimators])
