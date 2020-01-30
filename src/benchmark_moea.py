import datetime
import time
import random
import string
import os

import pygmo as pg
import pandas as pd
import numpy as np

from sklearn.model_selection import ParameterGrid


def make_nd_pop(pro, x, y):
    nd_front = pg.fast_non_dominated_sorting(y)[0][0]
    nd_x = x[nd_front]
    nd_y = y[nd_front]
    t_pop = pg.population(pro)
    for i, p_vector in enumerate(nd_x):
        t_pop.push_back(x=p_vector, f=nd_y[i])
    return t_pop


def experiment(problem_name: str,
               prob_id: int,
               prob_dim: int,
               obj: int,
               pop_size: int,
               gen: int,
               algo_name='nsga2',
               seed=None):

    result = {
        "problem_name": problem_name,
        "seed": seed,
        "problem_id": prob_id,
        "objectives": obj,
        "feature_dim": problem_name,
        'algo_name': algo_name,
        "pop_size": pop_size,
        "generation": gen,
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
    try:
        pop = pg.population(prob=udp, size=pop_size, seed=seed)
    except Exception as err:
        result['error'] = "Init population: {}".format(err)
        return result

    # ----------------------                                                            Initialization algorithm
    try:
        algo = pg.algorithm(getattr(pg, algo_name)(gen=gen, seed=seed))
    except Exception as err:
        result['error'] = "Init algorithm: {}".format(err)
        return result

    # ----------------------                                                            Solving
    evolve_start = time.time()
    try:
        pop = algo.evolve(pop)
        result["fevals"] = pop.problem.get_fevals()
        #This returns the first (i.e., best) non-dominated individual from population:
        nd_pop = make_nd_pop(prob, pop.get_x(), pop.get_f())

        score = udp.p_distance(nd_pop) if hasattr(udp, 'p_distance') else None
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
    #     The spacing metric aims at assessing the spread (distribution)
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

        result["problem_name"] = pop.problem.get_name()
        result["objectives"] = pop.problem.get_nobj()
        result["feature_dim"] = pop.problem.get_nx()
        result["pop_ndf_x"] = nd_pop.get_x()
        result["pop_ndf_f"] = nd_pop.get_f()
        result["evolve_time"] = t_end - evolve_start
        result["total_time"] = t_end - t_start
        result["date"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        result["final"] = True
        result['algo_name'] = algo_name
        result["p_distance"] = score or None
        result["ndf_size"] = len(nd_pop.get_f())

    except Exception as err:
        result['error'] = "Write results: {}".format(err)

    return result


if __name__ == "__main__":
    print(" Start MOEA benchamrk on multy-objective problems ")

    test_small = [
        {
            'problem_name': ['wfg'],
            'prob_id': [4],
            'prob_dim': [2],
            'obj': [2],
            'pop_size': [100],
            'gen': [200],
            'algo_name': ['moead', 'nsga2', 'maco', 'nspso'],
            'seed': [42]
        }
    ]

    i_total = 0
    for param_grid in test_small:
        i = 0
        res = []
        for p in ParameterGrid(param_grid):
            i = i+1
            print("\n Evaluation.: {} \n i: {}".format(p, i))
            res.append(experiment(**p))

        i_total = i_total + i

        # File and path to folder
        prefix = ''.join(random.choices(
            string.ascii_lowercase + string.digits, k=10))
        file_name = '/benchmark_results/MOEA_on_{}_i{}.{}.csv'.format(
            param_grid['problem_name'][0], i, prefix)
        path = os.path.dirname(os.path.abspath(__file__)) + file_name

        # Write results
        print("\n Total evaluations: {}".format(i_total))
        print(" Write results. Path:{} \n".format(path))
        res_table = pd.DataFrame(res)
        res_table.to_csv(path, mode='a+', index=False)
