import datetime
import time
import random
import string
import os
import logging
logging.basicConfig(filename='./bench_moea.log', level=logging.INFO)


from pprint import pformat

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
               algo_name,
               seed=None):

    result = {
        "problem_name": problem_name,
        "seed": seed,
        "problem_id": prob_id,
        "objectives": obj,
        "feature_dim": prob_dim,
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

    # ----------------------                                                            Initialization algorithm
    try:
        algo = pg.algorithm(getattr(pg, algo_name)(gen=1, seed=seed))
    except Exception as err:
        result['error'] = "Init algorithm: {}".format(err)
        return result

    # ----------------------                                                            Solving
    evolve_start = time.time()
    try:
        # --- Iteration
        arr_pops = pd.DataFrame()
        pop = pg.population(prob, size=pop_size)
        for i in range(gen):
            pop = algo.evolve(pop)
            temp_f = pd.DataFrame(pop.get_f()).add_prefix('f_')
            temp_x = pd.DataFrame(pop.get_x()).add_prefix('x_')
            temp = pd.concat([temp_x, temp_f], axis=1)
            temp['i_gen'] = i+1
            temp['i_fevals'] = pop.problem.get_fevals()
            nd_pop = make_nd_pop(prob, pop.get_x(), pop.get_f())
            temp["p_distance"] = udp.p_distance(nd_pop) if hasattr(udp, 'p_distance') else None
            ref_point = pg.nadir(pop.get_f())
            temp['hypervolume'] = pg.hypervolume(nd_pop.get_f()
                                                 ).compute(ref_point) or None
            temp["ndf_size"] = len(nd_pop.get_f())

            # ndf_space
            dist = pg.crowding_distance(points=nd_pop.get_f())
            not_inf_dist = dist[np.isfinite(dist)]
            mean_dist = np.mean(not_inf_dist)
            space_m = (sum([(mean_dist - d)**2 for d in not_inf_dist]
                        )/(len(not_inf_dist)-1))**(1/2)
            temp["ndf_space"] = space_m
            arr_pops = pd.concat([arr_pops, temp])

        # == File and path to folder
        prefix = ''.join(random.choices(
            string.ascii_lowercase + string.digits, k=4))
        file_name = '/benchmark_results/{}_on_{}_{}{}_gen{}.{}{}'.format(
            algo_name, 
            pop.problem.get_name(), 
            pop.problem.get_nx(), 
            pop.problem.get_nobj(), 
            gen, 
            seed,
            prefix)
        path = os.path.dirname(os.path.abspath(__file__)) + file_name
        # --- Write results
        # logging.info(" Write populations. Path:{} \n".format(path+'.csv'))
        # arr_pops.to_csv(path + '.csv', mode='a+', index=False)
        print(" Write populations. Path:{}".format(path + '.pkl'))
        arr_pops.to_pickle(path + '.pkl')


        result["fevals"] = pop.problem.get_fevals()
        nd_pop = make_nd_pop(prob, pop.get_x(), pop.get_f())
        score = udp.p_distance(nd_pop) if hasattr(udp, 'p_distance') else None
        result["p_distance"] = score
        result["evolve_time"] = time.time() - evolve_start
    except Exception as err:
        result['error'] = "Evolve: {}".format(err)
        return result

    # ----------------------                                                            Hypervolume
    try:
        ref_point = pg.nadir(pop.get_f())
        hypervolume = pg.hypervolume(nd_pop.get_f()
                                     ).compute(ref_point)
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
        result["problem_name"] = pop.problem.get_name()
        result["objectives"] = pop.problem.get_nobj()
        result["feature_dim"] = pop.problem.get_nx()
        result["pop_ndf_x"] = nd_pop.get_x().tolist()
        result["pop_ndf_f"] = nd_pop.get_f().tolist()
        result["total_time"] = time.time() - t_start
        result["date"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        result["final"] = True
        result['algo_name'] = algo_name
        result["ndf_size"] = len(nd_pop.get_f())

    except Exception as err:
        result['error'] = "Write results: {}".format(err)

    return result


if __name__ == "__main__":
    logging.info("----\n Start complex MOEA benchamrk on multy-objective problems\n ---- ")

    print("Start")

    test_set = [
        # {
        #     'problem_name': ['zdt'],
        #     'prob_id': [4],
        #     'prob_dim': [2],
        #     'obj': [2],
        #     'pop_size': [100],
        #     'gen': [500],
        #     'algo_name': ['moead', 'nsga2'],
        #     'seed': [303]
        # }
        {
            'problem_name': ['wfg'],
            'prob_id': [5, 6, 7, 8, 9],
            'prob_dim': [2],
            'obj': [2],
            'pop_size': [100],
            'gen': [500],
            'algo_name': ['moead', 'nsga2', 'maco', 'nspso'],
            'seed': [42, 72, 112, 774, 303]
        },
        {
            'problem_name': ['zdt'],
            'prob_id': [1, 2, 3, 5],
            'prob_dim': [2],
            'obj': [2],
            'pop_size': [100],
            'gen': [500],
            'algo_name': ['moead', 'nsga2', 'maco', 'nspso'],
            'seed': [42, 72, 112, 774, 303]
        },
        {
            'problem_name': ['dtlz'],
            'prob_id': [1, 2, 3, 5, 6, 7],
            'prob_dim': [3],
            'obj': [2],
            'pop_size': [100],
            'gen': [500],
            'algo_name': ['moead', 'nsga2', 'maco', 'nspso'],
            'seed': [42, 72, 112, 774, 303]
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
        file_name = '/benchmark_results/MOEA_{}_i{}.{}v3'.format(
            param_grid['problem_name'][0], total_comb, prefix)
        path = os.path.dirname(os.path.abspath(__file__)) + file_name

        # Write results
        logging.info("\n Total evaluations: {}".format(i_total))
        logging.info(" Write results. Path:{} \n".format(path + '.csv'))
        res_table = pd.DataFrame(res)
        res_table.to_csv(path + '.csv', mode='a+', index=False)

        print(" Write tutor sumary. Path:{}".format(path + '.pkl'))
        res_table.to_pickle(path + '.pkl')

    print("finish")
