import datetime
import time
import random
import string
import os
import logging


from pprint import pformat

import pygmo as pg
import pandas as pd
import numpy as np

from sklearn.model_selection import ParameterGrid

logging.basicConfig(filename='./temp_scal.log', level=logging.INFO)



def uniform_weights(dim: int = 2, total_sum=1):
    if dim == 1:
        return [total_sum]
    if total_sum <= 0:
        return [uniform_weights(dim-1, 0)]

    i_weight = random.uniform(0, total_sum)

    results = [i_weight] + uniform_weights(dim-1, total_sum - i_weight)
    return random.sample(results, len(results))

def make_nd_pop(pro, x, y):
    nd_front = pg.fast_non_dominated_sorting(y)[0][0]
    nd_x = np.array(x)[nd_front]
    nd_y = np.array(y)[nd_front]
    t_pop = pg.population(pro)
    for i, p_vector in enumerate(nd_x):
        t_pop.push_back(x=p_vector, f=nd_y[i])
    return t_pop


def experiment(problem_name: str,
               prob_id: int,
               prob_dim: int,
               obj: int,
               loop_size: int,
               scal_method: str,
               seed: int = None):

    result = {
        "problem_name": problem_name,
        "seed": seed,
        "problem_id": prob_id,
        "objectives": obj,
        "feature_dim": prob_dim,
        'scal_method': scal_method,
        "loop_size": loop_size,
        "pop_ndf_x": '',
        "pop_ndf_f": '',
        "fevals": '',
        "evolve_time": '',
        "total_time": '',
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

    t_start = time.time()
    # ----------------------                                                            Initial population
    # try:
    #     pop = pg.population(prob=udp, size=loop_size, seed=seed)
    # except Exception as err:
    #     result['error'] = "Init population: {}".format(err)
    #     return result

    # ----------------------                                                            Initialization algorithm
    try:
        algo = pg.algorithm(pg.simulated_annealing(
            Ts=10., Tf=1e-5, n_T_adj=100))
    except Exception as err:
        result['error'] = "Init algorithm: {}".format(err)
        return result

    # ----------------------                                                            Loop optimization
    evolve_start = time.time()
    try:
        islands = []
        # loop size defines how much Pareto-optimal points will be produced
        for _ in range(loop_size):
            scal_pro = pg.problem(pg.decompose(
                prob=prob, method=scal_method, weight=uniform_weights(obj, 1), z=[0.0]*obj))
            islands.append(pg.island(algo=algo, prob=scal_pro, size=1,
                            udi=pg.mp_island(), seed=seed))

        _ = [isl.evolve() for isl in islands]
        _ = [isl.wait() for isl in islands]

        x = [isl.get_population().champion_x for isl in islands]
        result["fevals"] = sum(
            [isl.get_population().problem.get_fevals() for isl in islands])
    except Exception as err:
        result['error'] = "Loop: {}".format(err)
        return result


    try:
        # Convert scaled predictions back to an original multi-objective problem
        y = [prob.fitness(p).tolist() for p in x]
        loop_pop = make_nd_pop(prob, x, y)

        # This returns the first (i.e., best) non-dominated individual from population:
        nd_pop = make_nd_pop(prob, loop_pop.get_x(), loop_pop.get_f())

        score = udp.p_distance(nd_pop) if hasattr(udp, 'p_distance') else None
        result["p_distance"] = score
        result["pop_ndf_x"] = nd_pop.get_x().tolist()
        result["pop_ndf_f"] = nd_pop.get_f().tolist()
        result["ndf_size"] = len(nd_pop.get_f())

    except Exception as err:
        result['error'] = "Non-dominated pop: {}".format(err)
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
    # of vectors throughout the set of non-dominated solutions
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
        result["total_time"] = t_end - t_start
        result["date"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        result["final"] = True

    except Exception as err:
        result['error'] = "Write results: {}".format(err)

    return result


if __name__ == "__main__":
    logging.info(
        "----\n Start Scalarization benchamrk on multy-objective problems\n ---- ")

    print("Start")

    test_set = [
        {
            'problem_name': ['wfg'],
            'prob_id': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'prob_dim': [2, 4, 6, 8, 10],
            'obj': [2, 4, 6, 8, 10],
            'loop_size': [40, 100, 200, 400, 800],
            'scal_method': ['bi', 'tchebycheff', 'weighted'],
            'seed': [42, 72, 112, 774, 111]
        },
        {
            'problem_name': ['zdt'],
            'prob_id': [1, 2, 3, 4, 5, 6],
            'prob_dim': [2, 4, 6, 8, 10],
            'obj': [2, 4, 6, 8, 10],
            'loop_size': [40, 100, 200, 400, 800],
            'scal_method': ['bi', 'tchebycheff', 'weighted'],
            'seed': [42, 72, 112, 774, 111]
        },
        {
            'problem_name': ['dtlz'],
            'prob_id': [1, 2, 3, 4, 5, 6, 7],
            'prob_dim': [2, 4, 6, 8, 10],
            'obj': [2, 4, 6, 8, 10],
            'loop_size': [40, 100, 200, 400, 800],
            'scal_method': ['bi', 'tchebycheff', 'weighted'],
            'seed': [42, 72, 112, 774, 111]
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
            res.append(experiment(**p))

        i_total = i_total + i

        # File and path to folder
        prefix = ''.join(random.choices(
            string.ascii_lowercase + string.digits, k=10))
        file_name = '/benchmark_results/scal_on_{}_i{}.{}.csv'.format(
            param_grid['problem_name'][0], i, prefix)
        path = os.path.dirname(os.path.abspath(__file__)) + file_name

        # Write results
        logging.info("\n Total evaluations: {}".format(i_total))
        logging.info(" Write results. Path:{} \n".format(path))
        res_table = pd.DataFrame(res)
        res_table.to_csv(path, mode='a+', index=False)

    print("finish")
