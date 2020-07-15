from typing import List, Tuple
import random

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

import pygmo as pg
import numpy as np

class Pygmo(BaseEstimator):
    """ massively parallel optimization
    """

    def __init__(self,
                 estimator=None,
                 bounds: Tuple[List] = None,
                 algo='nsga2',
                 **algo_params):
        self._estimator = estimator
        self._bounds = bounds
        self._problem = None

        cpp_inst = getattr(pg, algo)(**algo_params)
        self._algo = pg.algorithm(cpp_inst)
        self._algo_params = algo_params
        self._population = None

    def set_bounds(self, bounds) -> None:
        self._bounds = bounds

    def get_bounds(self) -> Tuple[List]:
        return self._bounds

    def _def_problem(self):
        problem = None
        if None not in (self._estimator, self._bounds):
            instance = CustomProblem(
                estimator=self._estimator,
                bounds=self._bounds,
                m_col=self._mask_col,
                m_value=self._mask_value
                )
            problem = pg.problem(instance)
        else:
            raise Exception(
                'Models and Bounds should not be None.\n Models:\n {} \n Bounds: {}'.format(self._estimators, self._bounds))
        return problem

    def _evolve(self):
        # mask insert as static values in problem
        self._problem = self._def_problem()
        
        isl = pg.island(algo=self._algo, 
                        prob=self._problem,
                        size=self._algo_params['size'])
        isl.evolve()
        isl.wait()
        e_pop = isl.get_population()

        if None not in (self._mask_col, self._mask_value):
            t_pop = pg.population(self._def_problem())
            evolve_x = e_pop.get_x()
            evolve_y = e_pop.get_f()
            for i, x_vector in enumerate(evolve_x):
                for c, v in zip(self._mask_col, self._mask_value):
                    x_vector = np.insert(x_vector, c, v, 0)
                t_pop.push_back(x=x_vector, f=evolve_y[i])
            self._population = t_pop
        else:
            self._population = e_pop

        print("Evolve {} by {}".format(
            self._problem.get_name(), self._algo))

    def transform(self, X, *arg, y=None, **kwargs):
        return X

    def fit(self, X, y=None, **kwargs):
        self._estimator = X
        self._evolve()
        return self

    def predict(self, *arg, X=None, count=-1, **kwargs):
        idx_ndf_front = pg.fast_non_dominated_sorting(
            self._population.get_f())[0][0]
        ndf_pop_x = self._population.get_x()[idx_ndf_front]

        if count > self._pop_size or count == -1:
            return ndf_pop_x
        elif count <= self._pop_size:
            return random.choices(ndf_pop_x, k=count)
        else:
            return "Invalid request solution"

    def get_name(self):
        return "pygmo.{} on {}".format(self._algo.get_name(), self._problem and self._problem.get_name())

    def score(self, X=None, y=None, sample_weight=None):
        try:
            ref_point = pg.nadir(self._population.get_f())
            hv = pg.hypervolume(self._population.get_f()).compute(ref_point)
        except ValueError as err:
            # print("Error: Negativ surrogate objectives")
            hv = None
        return hv


class CustomProblem():
    def __init__(self,
                 estimator,
                 bounds,
                 m_col=None,
                 m_value=None):
        check_is_fitted(estimator)
        self._estimator = estimator
        self._eval_method = 'predict'
        self._bounds = bounds
        self._mask_columns = m_col
        self._mask_value = m_value

    def set_mask(self, col=None, value=None):
        # if len(col) != len(value):
        #     raise ValueError(
        #         f"Columns and values should be equal length. Columns: {col}, values: {value}")
        self._mask_columns = col
        self._mask_value = value

    def get_mask(self, col, value):
        return (self.mask_columns, self.mask_value)

    def fitness(self, x):
        x = np.array(x)
        if None not in (self._mask_columns, self._mask_value):
            for c, v in zip(self._mask_columns, self._mask_value):
                x = np.insert(x, c, v, 0)

        result = getattr(self._estimator, self._eval_method)(x.reshape(1, -1))
        return result.tolist()[0]

    def get_nobj(self):
        prediction = self._estimators.predict([self._bounds[0]])
        nobj = len(prediction[0])

        return nobj

    # def get_nix(self):
    #     return len(self._bounds[0])

    # Return bounds of decision variables
    def get_bounds(self):
        if None not in (self._mask_value, self._mask_value):
            return tuple(np.delete(b, self._mask_columns, 0).flatten() for b in self._bounds)
        else:
            return self._bounds

    def set_bounds(self, bounds: Tuple[List]):
        self._bounds = bounds
        return self

    # Return function name
    def get_name(self):
        meta_name = type(self._estimator).__name__
        if hasattr(self._estimator, 'estimators_'):
            return meta_name + ": " + " vs ".join([type(t).__name__ for t in self._estimator.estimators_])
        else:
            return meta_name


def make_nd_pop(pro, x, y):
    nd_front = pg.fast_non_dominated_sorting(y)[0][0]
    nd_x = np.array(x)[nd_front]
    nd_y = np.array(y)[nd_front]
    t_pop = pg.population(pro)
    for i, p_vector in enumerate(nd_x):
        t_pop.push_back(x=p_vector, f=nd_y[i])
    return t_pop
