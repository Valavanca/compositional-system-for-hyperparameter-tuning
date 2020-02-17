from random import uniform
from pprint import pformat
import datetime
import time
import random
import string
import os
import logging
logging.basicConfig(filename='./temp_rand.log', level=logging.INFO)

import pygmo as pg
import pandas as pd
import numpy as np
import sobol_seq

from sklearn.model_selection import ParameterGrid


def sobol_sample(bounds=([0., 0.], [5., 5.]), n=1):
    """ Sobol sampling

    Args:
        bounds (Tuple):  Tuple with lower and higher bound for each feature in objective space.
        Example: (([0., 0.]), ([2., 4.]))
        n (int, optional): Sample count. Defaults to 1.

    Returns:
        List: 2D list of point(s) from search space
    """
    n_dim = len(bounds[0])
    sb = sobol_seq.i4_sobol_generate(n_dim, n, 1)
    diff = [r-l for l, r in zip(*bounds)]
    left = [l for l, _ in zip(*bounds)]
    return (sb*diff+left).tolist()


def lh_sample(bounds=([0., 0.], [5., 5.]), n=1):
    """ Latin Hypercube sampling

    Args:
        bounds (Tuple):  Tuple with lower and higher bound for each feature in objective space.
        Example: (([0., 0.]), ([2., 4.]))
        n (int, optional): Sample count. Defaults to 1.

    Returns:
        List: 2D list of point(s) from search space
    """
    n_dim = len(bounds[0])
    h_cube = np.random.uniform(size=[n, n_dim])
    for i in range(0, n_dim):
        h_cube[:, i] = (np.argsort(h_cube[:, i])+0.5)/n
    diff = [r-l for l, r in zip(*bounds)]
    left = [l for l, _ in zip(*bounds)]
    return (h_cube*diff+left).tolist()


def random_sample(bounds=([0., 0.], [5., 5.]), n=1):
    """ Random sampling

    Args:
        bounds (Tuple):  Tuple with lower and higher bound for each feature in objective space.
        Example: (([0., 0.]), ([2., 4.]))
        n (int, optional): Sample count. Defaults to 1.

    Returns:
        List: 2D list of point(s) from search space
    """
    v_uniform = np.vectorize(uniform)
    return [v_uniform(*bounds).tolist() for _ in range(n)]


def make_nd_pop(pro, x, y):
    nd_front = pg.fast_non_dominated_sorting(y)[0][0]
    nd_x = np.array(x)[nd_front]
    nd_y = np.array(y)[nd_front]
    t_pop = pg.population(pro)
    for i, p_vector in enumerate(nd_x):
        t_pop.push_back(x=p_vector, f=nd_y[i])
    return t_pop


def experiment_random(problem_name: str,
               prob_id: int,
               prob_dim: int,
               obj: int,
               random_points: int,
               sampling_plan: str,
               seed=None):

    np.random.seed(seed)

    result = {
        "problem_name": problem_name,
        "seed": seed,
        "problem_id": prob_id,
        "objectives": obj,
        "feature_dim": prob_dim,
        "random_points": random_points,
        "sampling_plan": sampling_plan,
        "ndf_x": '',
        "ndf_f": '',
        "fevals": '',
        "evolve_time": '',
        "date": '',
        "p_distance": '',
        "hypervolume": '',
        "ndf_space": '',
        "ndf_size": '',
        'error': '',
        'final': False
    }

    # ----------------------                                                            Initialize problem
    try:
        if problem_name is 'wfg':
            udp = pg.wfg(prob_id=prob_id, dim_dvs=prob_dim,
                         dim_obj=obj, dim_k=obj-1)
        elif problem_name is 'zdt':
            udp = pg.zdt(prob_id=prob_id, param=prob_dim)
        elif problem_name is 'dtlz':
            udp = pg.dtlz(prob_id=prob_id, dim=prob_dim, fdim=obj)
        prob = pg.problem(udp)
    except Exception as err:
        result['error'] = "Init problem: {}".format(err)
        return result

    # ----------------------                                                            Sampling plan initialization
    try:
        points = None
        if sampling_plan is 'random':
            points = random_sample(bounds=prob.get_bounds(), n=random_points)
        elif sampling_plan is 'sobol':
            points = sobol_sample(bounds=prob.get_bounds(), n=random_points)
        elif sampling_plan is 'latin':
            points = lh_sample(bounds=prob.get_bounds(), n=random_points)
        else:
            raise ValueError(
                f"{sampling_plan}. Parameter error! Acceptable values: 'random','sobol','latin'")
    except Exception as err:
        result['error'] = "Sampling plan initialization: {}".format(err)
        return result

    # ----------------------                                                            Solving
    evolve_start = time.time()
    try:
        points_f = pg.bfe()(prob, np.array(points).flatten()).reshape(-1, obj).tolist()

        result["fevals"] = prob.get_fevals()
        #This returns the first (i.e., best) non-dominated individual from population:
        nd_pop = make_nd_pop(prob, points, points_f)

        score = udp.p_distance(nd_pop) if hasattr(udp, 'p_distance') else None

        result["p_distance"] = score or None
        result["ndf_x"] = nd_pop.get_x().tolist()
        result["ndf_f"] = nd_pop.get_f().tolist()
        result["ndf_size"] = len(nd_pop.get_f())
    except Exception as err:
        result['error'] = "Evolve: {}".format(err)
        return result

    # ----------------------                                                            Hypervolume
    try:
        hypervolume = pg.hypervolume(-nd_pop.get_f()
                                     ).compute([0]*nd_pop.problem.get_nobj())
        result['hypervolume'] = hypervolume or None
    except Exception as err:
        result['error'] = "Hypervolume: {}".format(err)
        return result

    # ----------------------                                                            Spacing metric
    # The spacing metric aims at assessing the spread (distribution)
    # of vectors throughout the set of nondominated solutions.
    try:
        dist = pg.crowding_distance(points=nd_pop.get_f())
        not_inf_dist = dist[np.isfinite(dist)]
        mean_dist = np.mean(not_inf_dist)
        space_m = (sum([(mean_dist - d)**2 for d in not_inf_dist]
                       )/(len(not_inf_dist)-1))**(1/2)
        result["ndf_space"] = space_m
    except Exception as err:
        result['error'] = "Spacing metric: {}".format(err)
        return result

    # ----------------------                                                            Write results

    try:
        t_end = time.time()

        result["problem_name"] = nd_pop.problem.get_name()
        result["objectives"] = nd_pop.problem.get_nobj()
        result["feature_dim"] = nd_pop.problem.get_nx()
        result["evolve_time"] = t_end - evolve_start
        result["date"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        result["final"] = True
        result["sampling_plan"] = sampling_plan

    except Exception as err:
        result['error'] = "Write results: {}".format(err)

    return result


if __name__ == "__main__":
    logging.info(
        "----\n Start Random benchamrk on multy-objective problems\n ---- ")
    print("Start")

    test_set = [
        {
            'problem_name': ['wfg'],
            'prob_id': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'prob_dim': [2, 4, 6, 8, 10],
            'obj': [2, 4, 6, 8, 10],
            'random_points': [40, 100, 200, 400, 800, 5000, 10000],
            'sampling_plan': ['sobol', 'random', 'latin'],
            'seed': [42, 72, 112, 774]
        },
        {
            'problem_name': ['zdt'],
            'prob_id': [1, 2, 3, 4, 5, 6],
            'prob_dim': [2, 4, 6, 8, 10],
            'obj': [2, 4, 6, 8, 10],
            'random_points': [40, 100, 200, 400, 800, 5000, 10000],
            'sampling_plan': ['sobol', 'random', 'latin'],
            'seed': [42, 72, 112, 774]
        },
        {
            'problem_name': ['dtlz'],
            'prob_id': [1, 2, 3, 4, 5, 6, 7],
            'prob_dim': [2, 4, 6, 8, 10],
            'obj': [2, 4, 6, 8, 10],
            'random_points': [40, 100, 200, 400, 800, 5000, 10000],
            'sampling_plan': ['sobol', 'random', 'latin'],
            'seed': [42, 72, 112, 774]
        }
    ]

    logging.info(pformat(test_set))

    i_total = 0
    for param_grid in test_set:
        grid = ParameterGrid(param_grid)
        total_comb = len(grid)

        i = 0
        res = []
        for p in grid:
            i = i+1
            logging.info(
                "\n Evaluation.: {} \n i: {} from {}".format(p, i, total_comb))
            res.append(experiment_random(**p))

        i_total = i_total + i

        # File and path to folder
        prefix = ''.join(random.choices(
            string.ascii_lowercase + string.digits, k=10))
        file_name = '/benchmark_results/Random_on_{}_i{}.{}.csv'.format(
            param_grid['problem_name'][0], i, prefix)
        path = os.path.dirname(os.path.abspath(__file__)) + file_name

        # Write results
        logging.info("\n Total evaluations: %s", i_total)
        logging.info(" Write results. Path: %s \n", path)
        res_table = pd.DataFrame(res)
        res_table.to_csv(path, mode='a+', index=False)

    print("Finish")
