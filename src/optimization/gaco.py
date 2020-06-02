
from typing import List, Tuple
import random

from sklearn.base import BaseEstimator
import pygmo as pg
import numpy as np

from .abs_solver import Solver
from .share import Pagmo_problem

DEFAULT_POP_SIZE = 100
DEFAULT_GENERATION = 100


class Gaco(Solver):
    def __init__(self,
                 models: List[BaseEstimator] = None,
                 bounds: Tuple[List] = None,
                 mask_col: List[int] = None,
                 mask_val: List[int] = None,
                 pop_size=DEFAULT_POP_SIZE,
                 gen=DEFAULT_GENERATION):
        super(Gaco, self).__init__(models)
        self._pop_size = pop_size
        self._gen = gen
        self._bounds = bounds
        self._score_ref_point = None
        self._population = None
        self._problem = None
        # dimensions mask
        self._mask_col = mask_col,
        self._mask_value = mask_val

    @property
    def problem(self):
        return self._problem

    @property
    def pop_size(self):
        return self._pop_size

    @pop_size.setter
    def pop_size(self, count: int):
        self._pop_size = count if count > 0 else None

    @property
    def gen(self):
        return self._gen

    @gen.setter
    def gen(self, gen_number: int):
        self._gen = gen_number if gen_number > 0 else None

    @property
    def population(self):
        return self._population

    def set_mask(self, columns: List[int], values: List[int]) -> None:
        self._mask_col = columns
        self._mask_value = values
        self.__evolve()

    def get_mask(self):
        return self._mask_col, self._mask_value

    def set_bounds(self, bounds) -> None:
        self._bounds = bounds

    def get_bounds(self) -> Tuple[List]:
        return self._bounds

    def __def_problem(self, is_mask=False):
        problem = None
        if self._estimators and self._bounds is not None:
            instance = Pagmo_problem(
                models=self._estimators,
                bounds=self._bounds)
            if is_mask:
                instance.set_mask(self._mask_col, self._mask_value)
            problem = pg.problem(instance)
        else:
            raise Exception(
                'Models and Bounds should not be None.\n Models:\n {} \n Bounds: {}'.format(self._estimators, self._bounds))
        return problem

    def __evolve(self):
        self._problem = self.__def_problem(is_mask=True)
        algo = pg.algorithm(pg.nsga2(gen=self._gen))
        isl = pg.island(algo=algo, prob=self._problem, size=self._pop_size)
        isl.evolve()
        isl.wait()
        e_pop = isl.get_population()

        if None not in (self._mask_col, self._mask_value):
            t_pop = pg.population(self.__def_problem(is_mask=False))
            evolve_x = e_pop.get_x()
            evolve_y = e_pop.get_f()
            for i, x_vector in enumerate(evolve_x):
                for c, v in zip(self._mask_col, self._mask_value):
                    x_vector = np.insert(x_vector, c, v, 0)
                t_pop.push_back(x=x_vector, f=evolve_y[i])
            self._population = t_pop
        else:
            self._population = e_pop

        print("GACO: Evolve {} by {} population size in {} generation".format(
            self._problem.get_name(), self._pop_size, self._gen))

    def transform(self, X, *arg, y=None, **kwargs):
        return X

    def fit(self, X: List[BaseEstimator], y=None, **kwargs):
        self._estimator = X
        self.__evolve()
        return self 

    def predict(self, *arg, X=None, count=1, **kwargs):
        if self._population is None:
            self.__evolve()

        prediction = None
        if count > self._pop_size or count == -1:
            sort_index = np.argsort(self._population.get_f().flatten())[::-1]
            prediction = self._population.get_x()[sort_index]
        elif count <= self._pop_size:
            n_best_index = np.argsort(self._population.get_f().flatten())[::-1][:count]
            prediction = self._population.get_x()[n_best_index]
        else:
            raise Exception("Invalid request solution")

        return prediction

    def get_name(self):
        return "NSGA2: " + self._problem.get_name() if self._problem else 'None'

    def score(self, X=None, y=None, sample_weight=None):
        return self._population.champion_f[0]
