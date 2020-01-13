
from typing import List, Tuple
import random

from sklearn.base import BaseEstimator
import pygmo as pg
import numpy as np

from .abs_solver import Solver
from .share import Pagmo_problem

DEFAULT_POP_SIZE = 80
DEFAULT_GENERATION = 50


class Gaco(Solver):
    def __init__(self,
                 models: List[BaseEstimator] = None,
                 bounds: Tuple[List] = None,
                 pop_size=DEFAULT_POP_SIZE,
                 gen=DEFAULT_GENERATION):
        super(Gaco, self).__init__(models)
        self._pop_size = pop_size
        self._gen = gen
        self._bounds = bounds
        self._score_ref_point = None
        self._population = None
        self._problem = None

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

    def set_bounds(self, bounds) -> None:
        self._bounds = bounds

    def get_bounds(self) -> Tuple[List]:
        return self._bounds

    def __def_problem(self):
        if self.estimator and self._bounds is not None:
            instance = Pagmo_problem(
                models=self.estimator,
                is_single=True,
                bounds=self._bounds)
            self._problem = pg.problem(instance)
        else:
            raise Exception(
                'Models and Estimators should not be None.\n Models:\n {} \n Bounds: {}'.format(self.estimator, self._bounds))
        return self._problem

    def __evolve(self):
        self.__def_problem()
        init_pop = pg.population(self._problem, size=self._pop_size)
        # self._population = init_pop

        # old_score = self.score()
        algo = pg.algorithm(pg.gaco(gen=self._gen))
        algo.set_verbosity(200)
        self._population = algo.evolve(init_pop)
        # score = self.score()
        # print("Hypervolume: {}, Delta: {}".format(score, score-old_score))

        print("Gaco: Evolve {} by {} population size in {} generation".format(
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
