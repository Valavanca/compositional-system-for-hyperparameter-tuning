from typing import List, Tuple
import random

import pygmo as pg
import numpy as np

from .optimizer_abs import Optimizer
from .utilities import make_nd_pop


class RandOpt(Optimizer):
    """ random search optimal points
    """

    def __init__(self, algo: str = 'lhs'):
        super().__init__(method=algo)

    def minimize(self, surrogate, bounds, init= 100, **min_params):
        sol_x = self.generate_rand_solutions(bounds, count=init)
        sol_f = [surrogate.predict(x) for x in sol_x]
        sol_x_opt = []

        return sol_x_opt


    def generate_rand_solutions(self, bounds, count):
        samples = []
        if self._method == 'rand':
            rand_pop = pg.population(self._problem, size=self._n)
            pop = make_nd_pop(
                self._problem, rand_pop.get_x(), rand_pop.get_f())
        elif self._method == 'lhs':
            pop_x = lh_sample(self._bounds, self._n)
            pop_f = [self._problem.fitness(x).tolist() for x in pop_x]
            pop = make_nd_pop(self._problem, pop_x, pop_f)
        return samples

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
