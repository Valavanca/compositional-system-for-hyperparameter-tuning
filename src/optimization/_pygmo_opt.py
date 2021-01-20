from typing import List, Tuple
import random

from .optimizer_abs import Optimizer
# from sklearn.base import BaseEstimator
# from sklearn.utils.validation import check_is_fitted

import pygmo as pg
import numpy as np

class Pygmo(Optimizer):
    """ Wrapper for algorithms in pygmo: massively parallel optimization framework

    Reference:
        pygmo: https://esa.github.io/pygmo2/
    """

    def __init__(self,
                 algo='nsga2',
                 pop_size=None,
                 **algo_params):
        super().__init__(method=algo)
        cpp_inst = getattr(pg, algo)(**algo_params)
        self._algo = pg.algorithm(cpp_inst)
        self._pop_size = pop_size
        self._algo_params = algo_params

    def minimize(self, surrogate, bounds, mask_col=None, mask_value=None, **min_params):
        min_problem = self._create_problem(surrogate, bounds, mask_col, mask_value)

        isl = pg.island(algo=self._algo,
                        size=self._pop_size,
                        prob=min_problem,
                        udi=pg.mp_island(),
                        )
        isl.evolve()
        isl.wait()
        final_pop = isl.get_population()

        # pop = pg.population(prob=self._problem, size=self._pop_size)
        # e_pop = self._algo.evolve(pop)

        if None not in (mask_col, mask_value):
            t_pop = pg.population(min_problem)  # empty population
            evolve_x = t_pop.get_x()
            evolve_y = t_pop.get_f()
            for i, x_vector in enumerate(evolve_x):
                for c, v in zip(mask_col, mask_value):
                    x_vector = np.insert(x_vector, c, v, 0)
                t_pop.push_back(x=x_vector, f=evolve_y[i])
            initial_pop = t_pop
        else:
            final_pop

        print("Minimization {} by {}".format(
            min_problem.get_name(), self._algo))

        return self._get_solutions(final_pop)

    def _create_problem(self, surrogate, bounds, mask_col=None, mask_value=None):
        problem = None
        if None not in (surrogate, bounds):
            instance = PygmoProblem(
                surrogate=surrogate,
                bounds=bounds,
                m_col=mask_col,
                m_value=mask_value
            )
            problem = pg.problem(instance)
        else:
            raise Exception(
                'Estimator and Bounds should not be None.\n Estimator:\n {} \n Bounds: {}'.format(self._estimator, self._bounds))
        return problem

    def _get_solutions(self, population, count=-1, **kwargs):

        # --- 1. Select optimal vectors
        if len(population) > 1:
            idx_ndf_front = pg.fast_non_dominated_sorting(population.get_f())[0][0]  # index of final Pareto front
            ndf_pop_x = population.get_x()[idx_ndf_front]  # values of final Pareto front
        else:
            ndf_pop_x = population.get_x()

        # --- 2. Return require count of solutions
        if count == -1 or count > len(population):
            return ndf_pop_x
        elif count <= len(population):
            return random.choices(ndf_pop_x, k=count)
        else:
            return "Invalid request solution"

    def get_name(self):
        params = ", ".join("{}: {}".format(k, v) for k, v in self._algo_params.items())
        return "pygmo.{}({})".format(self._algo.get_name(), params)


class PygmoProblem():
    """ wrapper for an surrogate to use it as a problem for pygmo algorithms.
        Mask it is static values for some features(dimensions) in feature vectors. This means that optimization for these features is not carried out.
    """

    def __init__(self,
                 surrogate,
                 bounds,
                 eval_method='predict',
                 m_col=None,
                 m_value=None):
        self._surrogate = surrogate
        self._bounds = bounds
        self._eval_method = eval_method
        self._mask_columns = m_col
        self._mask_value = m_value

    def fitness(self, x):
        x = np.array(x)
        if None not in (self._mask_columns, self._mask_value):
            # recombine fitness vector with static values (mask columns and values)
            for c, v in zip(self._mask_columns, self._mask_value):
                x = np.insert(x, c, v, 0)

        result = getattr(self._surrogate, self._eval_method)(x.reshape(1, -1))
        return result.flatten().tolist()

    def get_nobj(self):
        prediction = self._surrogate.predict([self._bounds[0]])
        nobj = prediction.flatten().size
        return nobj

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
        meta_name = type(self._surrogate).__name__
        if hasattr(self._surrogate, 'estimators_'):
            return meta_name + ": " + " vs ".join([type(t).__name__ for t in self._surrogate.estimators_])
        else:
            return meta_name


# def make_nd_pop(pro, x, y):
#     nd_front = pg.fast_non_dominated_sorting(y)[0][0]
#     nd_x = np.array(x)[nd_front]
#     nd_y = np.array(y)[nd_front]
#     t_pop = pg.population(pro)
#     for i, p_vector in enumerate(nd_x):
#         t_pop.push_back(x=p_vector, f=nd_y[i])
#     return t_pop
