import datetime
import time
import random
import string

import pygmo as pg
import pandas as pd

from sklearn.model_selection import ParameterGrid


# ======================================================================


def experiment(problem_name: str,
               prob_id: int,
               prob_dim: int,
               obj: int,
               pop_size: int,
               gen: int,
               algo_name='nsga2',
               seed=None):

    result = {
        "problem_name": '',
        "seed": seed,
        "problem_id": prob_id,
        "objectives": '',
        "feature_dim": '',
        "pop_size": pop_size,
        "generation": gen,
        "pop_x": '',
        "pop_f": '',
        "evolve_time": '',
        "total_time": '',
        "date": '',
        "p_distance": '',
        "hypervolume": '',
        'error': '',
        'algo_name': '',
        'final': False
    }

    # ----------------------                                                            Make problem
    try:
        if problem_name is 'wfg':
            udp = pg.wfg(prob_id=prob_id, dim_dvs=prob_dim,
                         dim_obj=obj, dim_k=obj-1)
        elif problem_name is 'zdt':
            udp = pg.zdt(prob_id=prob_id, param=prob_dim)
        elif problem_name is 'dtlz':
            udp = pg.dtlz(prob_id=prob_id, dim=prob_dim, fdim=obj)
        pro = pg.problem(udp)
    except Exception as err:
        result['error'] = "Init problem: {}".format(err)
        return result

    # is_stochastic alias for has_set_seed()
    t_start = time.time()
    # ----------------------                                                            Initial population
    try:
        pop = pg.population(prob=udp, size=pop_size, seed=seed)
    except Exception as err:
        result['error'] = "Init population: {}".format(err)
        return result

    # ----------------------                                                            Algorithm initialization
    try:
        algo = pg.algorithm(getattr(pg, algo_name)(gen=gen))
    except Exception as err:
        result['error'] = "Init algorithm: {}".format(err)
        return result

    # ----------------------                                                            Solving
    evolve_start = time.time()
    try:
        pop = algo.evolve(pop)
        score = udp.p_distance(pop) if hasattr(udp, 'p_distance') else None
    except Exception as err:
        result['error'] = "Evolve: {}".format(err)
        return result

    # ----------------------                                                            Hypervolume
    try:
        hypervolume = pg.hypervolume(-pop.get_f()
                                     ).compute([0]*pop.problem.get_nobj())
    except Exception as err:
        result['error'] = "Hypervolume: {}".format(err)
        return result

    # ----------------------                                                            Write results
    try:
        t_end = time.time()

        result["problem_name"] = pop.problem.get_name()
        result["objectives"] = pop.problem.get_nobj()
        result["feature_dim"] = pop.problem.get_nx()
        result["pop_x"] = pop.get_x()
        result["pop_f"] = pop.get_f()
        result["evolve_time"] = t_end - evolve_start
        result["total_time"] = t_end - t_start
        result["date"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        result["final"] = True
        result['algo_name'] = algo_name
        result["p_distance"] = score or None
        result['hypervolume'] = hypervolume or None

    except Exception as err:
        result['error'] = "Write results: {}".format(err)

    return result


if __name__ == "__main__":
    print(" Start benchamrk ")

    # --- Benchmark config 

    test_suites_big = [
        {
            'problem_name': ['wfg'],
            'prob_id': [1, 2, 3, 4, 5, 6],
            'prob_dim': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            'obj': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            'pop_size': [20, 40, 80, 160, 400, 800, 1000],
            'gen': [20, 40, 80, 160, 400, 800, 1000],
            'algo_name': ['moead', 'nsga2', 'maco', 'nspso']
        },
        {
            'problem_name': ['zdt'],
            'prob_id': [1, 2, 3, 4, 5, 6, 7],
            'prob_dim': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            'obj': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            'pop_size': [20, 40, 80, 160, 400, 800, 1000],
            'gen': [20, 40, 80, 160, 400, 800, 1000],
            'algo_name': ['moead', 'nsga2', 'maco', 'nspso']
        },
    ]

    test_suites_small = [
        {
            'problem_name': ['zdt'],
            'prob_id': [2, 3],
            'prob_dim': [2, 3],
            'obj': [2, 3],
            'pop_size': [20, 40],
            'gen': [20, 40],
            'algo_name': ['moead', 'nsga2', 'maco', 'nspso']
        },
        {
            'problem_name': ['wfg'],
            'prob_id': [2, 3, 4],
            'prob_dim': [2, 3, 6, 10],
            'obj': [2, 3],
            'pop_size': [20, 40],
            'gen': [20, 40],
            'algo_name': ['moead', 'nsga2', 'maco', 'nspso']
        }
    ]

    #  config end
    #  --- 

    i_total = 0

    for param_grid in test_suites_big:
        i = 0
        res = []
        for p in ParameterGrid(param_grid):
            i = i+1
            print("\n Evaluation.: {} \n i: {}".format(p, i))
            res.append(experiment(**p))

        i_total = i_total + i
        prefix = ''.join(random.choices(
            string.ascii_lowercase + string.digits, k=10))

        path = './benchmark_results/{}_{}.csv'.format(
            param_grid['problem_name'][0], prefix)

        print(" Total evaluations: {}".format(i_total))
        print(" Write results. Path:{}".format(path))
        res_table = pd.DataFrame(res)
        res_table.to_csv(path, mode='a+', index=False)
