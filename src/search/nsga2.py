from typing import List, Tuple
import random

from sklearn.base import BaseEstimator
import pygmo as pg

from .abs_solver import Solver
from .share import Pagmo_problem
# from solid.tools import samples

DEFAULT_POP_SIZE = 80
DEFAULT_GENERATION = 80


class Nsga2(Solver):
    """ pygmo2 solver
    """

    def __init__(self,
                 models: List[BaseEstimator] = None,
                 bounds: Tuple[List] = None,
                 pop_size=DEFAULT_POP_SIZE,
                 gen=DEFAULT_GENERATION):
        super(Nsga2, self).__init__(models)
        self._estimators = models
        self._bounds = bounds
        self._problem = None
        self._pop_size = pop_size
        self._gen = gen
        self._population = None
        self._score_ref_point = None

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
        if self._estimators and self._bounds is not None:
            instance = Pagmo_problem(
                models=self._estimators,
                bounds=self._bounds)
            self._problem = pg.problem(instance)
        else:
            raise Exception(
                'Models and Bounds should not be None.\n Models:\n {} \n Bounds: {}'.format(self._estimators, self._bounds))
        return self._problem

    def __evolve(self):
        self.__def_problem()
        init_pop = pg.population(self._problem, size=self._pop_size)
        # self._score_ref_point = pg.hypervolume(init_pop).refpoint(offset=1)
        self._score_ref_point = pg.hypervolume(init_pop).refpoint(offset=1)
        self._population = init_pop

        # old_score = self.score()
        algo = pg.algorithm(pg.nsga2(gen=self._gen))
        algo.set_verbosity(200)
        self._population = algo.evolve(init_pop)
        # score = self.score()
        # print("Hypervolume: {}, Delta: {}".format(score, score-old_score))

        print("Evolve {} by {} population size in {} generation".format(
            self._problem.get_name(), self._pop_size, self._gen))

    def transform(self, X, *arg, y=None, **kwargs):
        return X

    def fit(self, X: List[BaseEstimator], y=None, **kwargs):
        self._estimators = X
        self.__evolve()
        return self

    def predict(self, *arg, X=None, count=-1, **kwargs):
        if count > self._pop_size or count == -1:
            return self._population.get_x()
        elif count <= self._pop_size:
            return random.choices(self._population.get_x(), k=count)
        else:
            return "Invalid request solution"

    def get_name(self):
        return "NSGA2: " + self._problem.get_name() if self._problem else 'None'

    def score(self, X=None, y=None, sample_weight=None):
        hv = pg.hypervolume(-self._population.get_f())
        return hv.compute([2]*self._population.get_f().shape[1])
