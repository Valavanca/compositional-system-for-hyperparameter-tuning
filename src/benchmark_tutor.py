import warnings
from hypothesis.custom_gp_kernel import KERNEL_MAUNA, KERNEL_SIMPLE, KERNEL_GPML
from generator import SamplesGenerator
from composite import PredictTutor, ModelsUnion
from joblib import Parallel, delayed
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import sklearn.gaussian_process as gp
from sklearn.model_selection import ParameterGrid
from sklearn import clone
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
logging.basicConfig(filename='./tutor_vs_sampling_plan.log', level=logging.INFO)


warnings.filterwarnings('ignore')


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


def make_nd_pop(pro, x, y):
    nd_front = pg.fast_non_dominated_sorting(y)[0][0]
    nd_x = np.array(x)[nd_front]
    nd_y = np.array(y)[nd_front]
    t_pop = pg.population(pro)
    for i, p_vector in enumerate(nd_x):
        t_pop.push_back(x=p_vector, f=nd_y[i])
    return t_pop


def get_static_ref_point(prob, offset=1):
    SEED = 214
    rand_pop = pg.population(prob, size=1000, seed=SEED)

    # moead
    algo = pg.algorithm(pg.moead(gen=300, seed=SEED))
    moead_pop = algo.evolve(pg.population(prob, size=100, seed=SEED))

    # NSGA 2
    algo = pg.algorithm(pg.nsga2(gen=300, seed=214))
    nsga_pop = algo.evolve(pg.population(prob, size=100, seed=SEED))

    # nspso
    algo = pg.algorithm(pg.nspso(gen=300, seed=214))
    nspso_pop = algo.evolve(pg.population(prob, size=100, seed=SEED))

    sum_pop_f = np.concatenate(
        (moead_pop.get_f(),
         nsga_pop.get_f(),
         nspso_pop.get_f(),
         rand_pop.get_f()),
        axis=0)

    return pg.nadir(sum_pop_f+offset)


def tuning_loop(pro,
                X_init, y_init,
                surr_portfolio,
                eval_budget,
                n_pred,
                solver,
                train_test_sp,
                cv_threshold,
                test_threshold,
                solution_comb):
    gen = SamplesGenerator(pro)
    if np.array(X_init).size > 0:
        gen.update(X_init, y_init)

    tutor = PredictTutor(pro.get_bounds(),
                         portfolio=surr_portfolio,
                         solver=solver,
                         train_test_sp=train_test_sp,
                         cv_threshold=cv_threshold,
                         test_threshold=test_threshold)

    loop_start = time.time()
    iter_solution = []
    i = 0

    n_iter = int(eval_budget/n_pred)
    while i < n_iter:
        i = i+1
        logging.info("\n--- {}".format(i))
        X, y = gen.return_X_y()
        tutor.fit(X, y, cv=4)
        propos = tutor.predict(n=n_pred, kind=solution_comb)
        logging.info(propos)

        pred = json.loads(tutor.predict_proba(
            None).to_json(orient='records'))[0]
        pred['prediction'] = propos.tolist()
        pred['iteration'] = i
        pred['problem'] = pro.get_name()
        pred['objectives'] = pro.get_nobj()
        pred['feature_dim'] = pro.get_nx()

        pred['eval_budget'] = eval_budget
        pred['n_pred'] = n_pred

        pred['solver'] = solver
        pred['samples_x'] = ''
        pred['samples_y'] = ''
        pred['solution_comb'] = solution_comb
        pred['train_test_sp'] = train_test_sp
        pred['cv_threshold'] = cv_threshold
        pred['test_threshold'] = test_threshold
        pred['init_dataset'] = np.array(X_init).size

        # Update dataset
        gen.update(list(propos), [pro.fitness(p).tolist() for p in propos])

        # ----------------------                                                             Hypervolume
        samples_x, samples_y = gen.return_X_y()
        pred['samples_x'] = samples_x
        pred['samples_y'] = samples_y
        if 0 in (np.array(samples_x).size, np.array(samples_y).size):
            continue

        try:
            # ref_point = pg.nadir(np.array(samples_y))
            ref_point = np.amax(np.array(samples_y), axis=0).tolist()
            pred['ref_point'] = ref_point
            nd_pop = make_nd_pop(pro, np.array(samples_x), np.array(samples_y))
            hypervolume = pg.hypervolume(nd_pop.get_f()
                                         ).compute(ref_point)
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
    rel_path = '/benchmark_results/{}_{}_plan_tutor_loop.{}.csv'.format(
        pro.get_name(), pro.get_nobj(), loop_prefix)
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
               pred_count: int,
               eval_budget: int,
               surr_port,

               solver: str,
               train_test_sp: float,
               cv_threshold: str,
               test_threshold: str,
               solution_comb: str,
               start_set: float,
               seed=None):

    result = {
        "problem_name": problem_name,
        "seed": seed,
        "problem_id": prob_id,
        "objectives": obj,
        "feature_dim": prob_dim,
        "pred_count": pred_count,

        'eval_budget': eval_budget,
        'surr_portfolio': surr_port,

        'solver': solver,
        'train_test_sp': train_test_sp,
        'cv_threshold': cv_threshold,
        'test_threshold': test_threshold,
        'solution_comb': solution_comb,
        'start_set': start_set,

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

    # ----------------------                                                            Initial sample plan
    try:
        start_x = []
        start_f = []
        if start_set > 0:
            n = int(eval_budget*start_set)
            eval_budget = eval_budget - n

            start_x = lh_sample(prob.get_bounds(), n)
            start_f = [prob.fitness(x).tolist() for x in start_x]
        else:
            pass

    except Exception as err:
        result['error'] = "Init sample plan: {}".format(err)
        return result


    # ----------------------                                                            Tutor model loop
    evolve_start = time.time()

    try:
        x_loop, y_loop = tuning_loop(prob,
                                     start_x, start_f,
                                     surr_port,
                                     eval_budget,
                                     pred_count,
                                     solver,
                                     train_test_sp,
                                     cv_threshold,
                                     test_threshold,
                                     solution_comb)

        result["fevals"] = prob.get_fevals()
        nd_pop = make_nd_pop(prob, x_loop, y_loop)
        score = udp.p_distance(nd_pop) if hasattr(udp, 'p_distance') else None
        result["p_distance"] = score or None
    except Exception as err:
        result['error'] = "Tutor loop: {}".format(err)
        return result

    # ----------------------                                                            Hypervolume
    try:
        # ref_point = pg.nadir(y_loop)
        ref_point = np.amax(y_loop, axis=0).tolist()

        hypervolume = pg.hypervolume(nd_pop.get_f()
                                     ).compute(ref_point)
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
        "----\n Start Sample plan benchamrk on multy-objective problems\n ---- ")

    print("Start")

    # 1
    # tea_pot = TpotWrp(generations=2, population_size=10)
    # 2
    gp_mauna = gp.GaussianProcessRegressor(
        kernel=KERNEL_MAUNA, n_restarts_optimizer=20)
    # 3
    grad_uni = ModelsUnion(
        models=[GradientBoostingRegressor(n_estimators=500)],
        split_y=True)
    # 4
    lin_uni = ModelsUnion(models=[LinearRegression()], split_y=True)

    # 5
    svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    svr_uni = ModelsUnion(models=[svr_rbf], split_y=True)

    # 6
    mlp_reg = MLPRegressor(hidden_layer_sizes=(
        20, 60, 20), activation='relu', solver='lbfgs')
    mlp_uni = ModelsUnion(models=[mlp_reg], split_y=True)


    test_set = [
        # {
        #     'problem_name': ['zdt'],
        #     'prob_id': [6],
        #     'prob_dim': [2],
        #     'obj': [2],
        #     'eval_budget': [1000],
        #     'pred_count': [50],
        #     'surr_port': [[gp_mauna, grad_uni, svr_uni, mlp_uni]],
        #     'solver': ['nsga2', 'moea_control', 'random'],
        #     'train_test_sp': [0.25, ],
        #     'cv_threshold': ['(test_r2 > 0.2)'],
        #     'test_threshold': ['(ndf_surr_score > 0.6)'],
        #     'solution_comb': ['stack'],
        #     'start_set': [0.1],
        #     'seed': [42]
        # }
        { 
            'problem_name': ['zdt'],
            'prob_id': [4, 6],
            'prob_dim': [2],
            'obj': [2],
            'eval_budget': [1000],
            'pred_count': [25, 50],
            'surr_port': [[gp_mauna, grad_uni, svr_uni, mlp_uni]],
            'solver': ['nsga2', 'moea_control', 'random'],
            'train_test_sp': [0.1, 0.25, 0.5],
            'cv_threshold': ['(test_r2 > 0.2)', '(test_r2 > 0.65)', '(test_r2 > 0.85)'],
            'test_threshold': ['(ndf_surr_score > 0)', '(ndf_surr_score > 0.6)', '(ndf_surr_score > 0.85)'],
            'solution_comb': ['ndf_score', 'stack'],
            'start_set': [0.1, 0.5, 0.75],
            'seed': [42]
        },
        {
            'problem_name': ['wfg'],
            'prob_id': [2, 4],
            'prob_dim': [2],
            'obj': [2],
            'eval_budget': [1000],
            'pred_count': [25, 50],
            'surr_port': [[gp_mauna, grad_uni, svr_uni, mlp_uni]],
            'solver': ['nsga2', 'moea_control', 'random'],
            'train_test_sp': [0.1, 0.25, 0.5],
            'cv_threshold': ['(test_r2 > 0.2)', '(test_r2 > 0.65)', '(test_r2 > 0.85)'],
            'test_threshold': ['(ndf_surr_score > 0)', '(ndf_surr_score > 0.6)', '(ndf_surr_score > 0.85)'],
            'solution_comb': ['ndf_score', 'stack'],
            'start_set': [0.1, 0.5, 0.75],
            'seed': [42]
        },
        {
            'problem_name': ['dtlz'],
            'prob_id': [3, 4, 6],
            'prob_dim': [3],
            'obj': [2],
            'eval_budget': [1000],
            'pred_count': [10, 25, 50],
            'surr_port': [[gp_mauna, grad_uni, svr_uni, mlp_uni]],
            'solver': ['nsga2', 'moea_control', 'random'],
            'train_test_sp': [0.1, 0.25, 0.5],
            'cv_threshold': ['(test_r2 > 0.2)', '(test_r2 > 0.65)', '(test_r2 > 0.85)'],
            'test_threshold': ['(ndf_surr_score > 0)', '(ndf_surr_score > 0.6)', '(ndf_surr_score > 0.85)'],
            'solution_comb': ['ndf_score', 'stack'],
            'start_set': [0.1, 0.5, 0.75],
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
            file_name = '/benchmark_results/sampler_tutor_on_{}_i{}.{}.csv'.format(
                param_grid['problem_name'][0], i, prefix)
            path = os.path.dirname(os.path.abspath(__file__)) + file_name

            # Write results
            # logging.info("\n Total evaluations: {}".format(i_total))
            logging.info(" Write results. Path:{} \n".format(path))
            res_table = pd.DataFrame(res)
            res_table.to_csv(path, mode='a+', index=False)

    print("Finish")