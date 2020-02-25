import numpy as np
import pandas as pd
import pygmo as pg
from pprint import pformat
import datetime
import time
import random
import string
import json
import os
import logging
logging.basicConfig(filename='./temp_tutor.log', level=logging.INFO)

from sklearn.model_selection import ParameterGrid
import sklearn.gaussian_process as gp
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor

from joblib import Parallel, delayed

from composite import PredictTutor, ModelsUnion
from generator import SamplesGenerator

from hypothesis.custom_gp_kernel import KERNEL_MAUNA, KERNEL_SIMPLE, KERNEL_GPML


import warnings
warnings.filterwarnings('ignore')

def make_nd_pop(pro, x, y):
    nd_front = pg.fast_non_dominated_sorting(y)[0][0]
    nd_x = np.array(x)[nd_front]
    nd_y = np.array(y)[nd_front]
    t_pop = pg.population(pro)
    for i, p_vector in enumerate(nd_x):
        t_pop.push_back(x=p_vector, f=nd_y[i])
    return t_pop


def tuning_loop(pro, surr_portfolio, n_iter, n_pred=1):
    gen = SamplesGenerator(pro)
    # tutor = make_pipeline(PredictTutor(pro.get_bounds(), portfolio=[grad_uni, lin_uni]))
    tutor = PredictTutor(pro.get_bounds(), portfolio=surr_portfolio)

    loop_start = time.time()
    iter_solution = []
    i = 0
    while i < n_iter:
        i = i+1
        logging.info("\n--- {}".format(i))
        X, y = gen.return_X_y()
        tutor.fit(X, y, cv=4)
        propos = tutor.predict(n_pred=n_pred)
        logging.info(propos)

        pred = json.loads(tutor.predict_proba(
            None).to_json(orient='records'))[0]
        pred['prediction'] = propos
        pred['iteration'] = i
        pred['problem'] = pro.get_name()
        pred['objectives'] = pro.get_nobj()
        pred['feature_dim'] = pro.get_nx()

        # Update dataset
        gen.update(list(propos), [pro.fitness(p).tolist() for p in propos])

        # ----------------------                                                             Hypervolume
        samples_x, samples_y = gen.return_X_y()
        if 0 in (np.array(samples_x).size, np.array(samples_y).size):
            continue

        try:
            nd_pop = make_nd_pop(pro, np.array(samples_x), np.array(samples_y))
            hypervolume = pg.hypervolume(-nd_pop.get_f()
                                         ).compute([0]*nd_pop.problem.get_nobj())
            pred['hypervolume'] = hypervolume or None
            pred["ndf_size"] = len(nd_pop.get_f())
        except Exception as err:
            pred['error'] = "Hypervolume: {}".format(err)
            iter_solution.append(pred)
            continue

        # ----------------------                                                            Spacing metric
        try:
            dist = pg.crowding_distance(points=nd_pop.get_f())
            not_inf_dist = dist[np.isfinite(dist)]
            mean_dist = np.mean(not_inf_dist)
            space_m = (sum([(mean_dist - d)**2 for d in not_inf_dist]
                           )/(len(not_inf_dist)-1))**(1/2)
            pred["ndf_space"] = space_m
        except Exception as err:
            pred['error'] = "Spacing metric: {}".format(err)
            iter_solution.append(pred)
            continue

        pred["i_time"] = time.time() - loop_start
        iter_solution.append(pred)

    loop = pd.DataFrame(iter_solution)
    loop = loop.drop(['estimator'], axis=1, errors='ignore')
    loop = loop.assign(tutor_id=id(tutor))

    # File and path to folder
    loop_prefix = loop.iloc[-1].tutor_id
    rel_path = '/benchmark_results/{}_tutor_loop_{}.{}.csv'.format(
        pro.get_name(), len(iter_solution), loop_prefix)
    path = os.path.dirname(os.path.abspath(__file__))

    # Write results
    print(" Write meta. Path:{}".format(path + rel_path))
    loop.to_csv(path + rel_path, mode='a+', index=False)

    X, y = gen.return_X_y()
    return np.array(X), np.array(y)


def experiment(problem_name: str,
               prob_id: int,
               prob_dim: int,
               obj: int,
               eval_budget: int,
               surr_port,
               seed=None):

    result = {
        "problem_name": problem_name,
        "seed": seed,
        "problem_id": prob_id,
        "objectives": obj,
        "feature_dim": prob_dim,

        'eval_budget': eval_budget,
        'surr_portfolio': surr_port,

        "pop_ndf_x": '',
        "pop_ndf_f": '',
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

    # ----------------------                                                            Tutor model loop
    evolve_start = time.time()
    try:
        x_loop, y_loop = tuning_loop(prob, surr_port, eval_budget)

        result["fevals"] = prob.get_fevals()
        nd_pop = make_nd_pop(prob, x_loop, y_loop)
        score = udp.p_distance(nd_pop) if hasattr(udp, 'p_distance') else None
        result["p_distance"] = score or None
    except Exception as err:
        result['error'] = "Tutor loop: {}".format(err)
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

        result["pop_ndf_x"] = nd_pop.get_x().tolist()
        result["pop_ndf_f"] = nd_pop.get_f().tolist()
        result["evolve_time"] = t_end - evolve_start
        result["date"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        result["final"] = True
        result["ndf_size"] = len(nd_pop.get_f())

    except Exception as err:
        result['error'] = "Write results: {}".format(err)

    return result


if __name__ == "__main__":
    logging.info(
        "----\n Start Tutor Model benchamrk on multy-objective problems\n ---- ")

    print("Start")

    # 1
    # tea_pot = TpotWrp(generations=2, population_size=10)
    # 2
    gp_sim = gp.GaussianProcessRegressor(
        kernel=KERNEL_SIMPLE, alpha=0, n_restarts_optimizer=10, normalize_y=True)
    # 3
    grad_uni = ModelsUnion(
        models=[GradientBoostingRegressor(n_estimators=200)],
        split_y=True)
    # 4
    lin_uni = ModelsUnion(models=[LinearRegression()], split_y=True)

    test_set = [
        {
            'problem_name': ['zdt'],
            'prob_id': [1, 2, 3, 4, 5, 6],
            'prob_dim': [2],
            'obj': [2],
            'eval_budget': [300],
            'surr_port': [[gp_sim, grad_uni, lin_uni]],
            'seed': [42]
        },
        {
            'problem_name': ['wfg'],
            'prob_id': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'prob_dim': [2],
            'obj': [2],
            'eval_budget': [300],
            'surr_port': [[gp_sim, grad_uni, lin_uni]],
            'seed': [42]
        },
        {
            'problem_name': ['dtlz'],
            'prob_id': [1, 2, 3, 4, 5, 6, 7],
            'prob_dim': [2, 4, 6, 8, 10],
            'obj': [2, 4, 6, 8, 10],
            'eval_budget': [300],
            'surr_port': [[gp_sim, grad_uni, lin_uni]],
            'seed': [42]
        }
    ]

    logging.info(pformat(test_set))

    i_total = 0
    with Parallel(prefer='threads') as parallel:
        for param_grid in test_set:
            grid = ParameterGrid(param_grid)
            total_comb = len(grid)
            logging.info(
                "\n Total combinations in round: {}".format(total_comb))

            i = 0
            # res = []
            # for p in grid:
            #     i = i+1
            #     logging.info(
            #         "\n Evaluation.: {} \n i: {} from {}".format(p, i, total_comb))
            #     res.append(experiment(**p))

            
            res = parallel(delayed(experiment)(**p) for p in grid)

            i_total = i_total + i

            # File and path to folder
            prefix = ''.join(random.choices(
                string.ascii_lowercase + string.digits, k=10))
            file_name = '/benchmark_results/mtutor_on_{}_i{}.{}.csv'.format(
                param_grid['problem_name'][0], i, prefix)
            path = os.path.dirname(os.path.abspath(__file__)) + file_name

            # Write results
            # logging.info("\n Total evaluations: {}".format(i_total))
            logging.info(" Write results. Path:{} \n".format(path))
            res_table = pd.DataFrame(res)
            res_table.to_csv(path, mode='a+', index=False)

    print("Finish")
