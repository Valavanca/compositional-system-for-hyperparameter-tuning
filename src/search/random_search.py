from typing import List, Tuple
import random

from sklearn.base import BaseEstimator
import pygmo as pg
import numpy as np

from .abs_solver import Solver
from .share import Pagmo_problem, make_nd_pop
# from solid.tools import samples


class RandS(Solver):
    """ pygmo2 solver
    """

    def __init__(self,
                 models: List[BaseEstimator] = None,
                 bounds: Tuple[List] = None,
                 n: int = 100,
                 kind: str = 'lhs',
                 mask_col: List[int] = None,
                 mask_val: List[int] = None):
        super(RandS, self).__init__(models)
        self._estimators = models
        self._bounds = bounds
        self._problem = None
        self._n = n
        self._kind = kind
        # dimensions mask
        self._mask_col = mask_col,
        self._mask_value = mask_val

    @property
    def problem(self):
        return self._problem

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
        pop = self.generate_rand_pop()

        if None not in (self._mask_col, self._mask_value):
            t_pop = pg.population(self.__def_problem(is_mask=False))
            evolve_x = pop.get_x()
            evolve_y = pop.get_f()
            for i, x_vector in enumerate(evolve_x):
                for c, v in zip(self._mask_col, self._mask_value):
                    x_vector = np.insert(x_vector, c, v, 0)
                t_pop.push_back(x=x_vector, f=evolve_y[i])
            self._population = t_pop
        else:
            self._population = pop

        print("Random evolve {} samples on a {} plan. None-dominated is {}".format(
            self._n, self._kind, len(self._population)))

    def generate_rand_pop(self):
        if self._kind == 'rand':
            rand_pop = pg.population(self._problem, size=self._n)
            pop = make_nd_pop(
                self._problem, rand_pop.get_x(), rand_pop.get_f())
        elif self._kind == 'lhs':
            pop_x = lh_sample(self._bounds, self._n)
            pop_f = [self._problem.fitness(x).tolist() for x in pop_x]
            pop = make_nd_pop(self._problem, pop_x, pop_f)
        return pop

    def transform(self, X, *arg, y=None, **kwargs):
        return X

    def fit(self, X: List[BaseEstimator], y=None, **kwargs):
        self._estimators = X
        self.__evolve()
        return self

    def predict(self, *arg, X=None, count=-1, **kwargs):
        idx_ndf_front = pg.fast_non_dominated_sorting(
            self._population.get_f())[0][0]
        ndf_pop_x = self._population.get_x()[idx_ndf_front]

        if count > len(self._population) or count == -1:
            return ndf_pop_x
        elif count <= len(self._population):
            return random.choices(ndf_pop_x, k=count)
        else:
            return "Invalid request solution"

    def get_name(self):
        return "Random search: " + self._problem.get_name() if self._problem else 'None'

    def score(self, X=None, y=None, sample_weight=None):
        try:
            ref_point = pg.nadir(self._population.get_f())
            hv = pg.hypervolume(self._population.get_f()).compute(ref_point)
        except ValueError as err:
            # print("Error: Negativ surrogate objectives")
            hv = None
        return hv


def lh_sample(bounds, n=1):
    """ Latin Hypercube sampling

    Args:
        bounds (Tuple):  Tuple with lower and higher bound for each feature in objective space.
        Example: (([0., 0.]), ([2., 4.]))
        n (int, optional): Sample count. Defaults to 1.

    Returns:
        List: Point from search space
    """
    n_dim = len(bounds[0])
    h_cube = np.random.uniform(size=[n, n_dim])
    for i in range(0, n_dim):
        h_cube[:, i] = (np.argsort(h_cube[:, i])+0.5)/n
    diff = [r-l for l, r in zip(*bounds)]
    left = [l for l, _ in zip(*bounds)]
    return h_cube*diff+left
