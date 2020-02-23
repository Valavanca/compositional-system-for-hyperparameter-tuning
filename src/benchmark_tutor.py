import datetime
import time
import random
import string
import os


# --- Dependencies
import pygmo as pg
import numpy as np
import pandas as pd

from sklearn.model_selection import ParameterGrid
from sklearn import  clone
import sklearn.gaussian_process as gp
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor

from composite import PredictTutor, ModelsUnion
from generator import SamplesGenerator

from hypothesis.custom_gp_kernel import KERNEL_MAUNA, KERNEL_SIMPLE, KERNEL_GPML


def experiment_tutor(problem_name: str,
                     portfolio,
                     prob_id: int,
                     prob_dim: int,
                     obj: int,
                     tuning_iter: int,
                     seed=None):

    meta_info = {
        "problem_name": problem_name,
        "seed": seed,
        "problem_id": prob_id,
        "objectives": obj,
        "feature_dim": prob_dim,
        "total_iterations": tuning_iter,
        "tutor_id": '',
        "loop_time": '',
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        'error': ''
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
        meta_info['error'] = "Init problem: {}".format(err)
        return meta_info, []

    t_start = time.time()
    # ----------------------                                                            Initial generator and tutor model
    try:
        gen = SamplesGenerator(pro)
        tutor = PredictTutor(pro.get_bounds(), portfolio=portfolio)
        tutor_id = id(tutor)
        meta_info['tutor_id'] = tutor_id
    except Exception as err:
        meta_info['error'] = "Init generator and tutor model: {}".format(err)
        return meta_info, []


    tutor_i_solution = []
    iter_colection = []
    i = 0
    # ----------------------                                                            Parameter tuning iterations
    try:
        loop_start = time.time()
        while i < tuning_iter:
            i = i+1
            iter_start = time.time()
            X, y = gen.return_X_y()

            iter_result = {
                "problem_name": problem_name,
                "seed": seed,
                "problem_id": prob_id,
                "objectives": obj,
                "feature_dim": prob_dim,
                "total_iterations": tuning_iter,
                "iteration": i,
                "samples_X": X.values.tolist() if isinstance(X, pd.DataFrame) else [],
                "samples_y": y.values.tolist() if isinstance(y, pd.DataFrame) else [],
                "tutor_id": tutor_id,
                "tutor_pred_count": '',
                "tutor_pred": '',
                "tutor_pred_fitness": '',
                "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                'error': ''
            }

            try:
                n_predictions = 1
                tutor.fit(X, y, cv=4)
                propos = tutor.predict(n=n_predictions)
                iter_result["tutor_pred"] = propos.tolist()
                iter_result["tutor_pred_count"] = n_predictions
            except Exception as err:
                iter_result['error'] = "Tutor. Get next config: {}".format(err)
                iter_colection.append(iter_result)
                continue

            # TODO Score metrics
            

            try:
                evaluation = [pro.fitness(p).tolist() for p in propos]
                gen.update(propos.tolist(), evaluation)
                iter_result["tutor_pred_fitness"] = evaluation
            except Exception as identifier:
                iter_result['error'] = "Update generator. {}".format(err)
                iter_colection.append(iter_result)
                continue

            iter_result["iter_time"] = time.time() - iter_start
            iter_colection.append(iter_result)

            temp_df_sol = tutor.solution
            temp_df_sol["tutor_id"] = tutor_id
            temp_df_sol["problem_name"] = problem_name
            temp_df_sol["objectives"] = obj
            temp_df_sol["feature_dim"] = prob_dim
            temp_df_sol["iteration"] = i
            tutor_i_solution.append(temp_df_sol)

        # Write tutor iterative solutions
        # File and path to folder
        prefix = ''.join(random.choices(
            string.ascii_lowercase + string.digits, k=10))
        file_name = '/benchmark_results/tutor_i_solution_on_{}({},{})_i{}.{}.csv'.format(
            problem_name, prob_dim, obj, len(tutor_i_solution), prefix)
        path = os.path.dirname(os.path.abspath(__file__)) + file_name
        # Write results
        logging.info(" Write restutor_i_solutions. Path:{} \n".format(path))
        pd.DataFrame(tutor_i_solution).to_csv(path, mode='a+', index=False)

        meta_info['loop_time'] = time.time() - loop_start
    except Exception as err:
        meta_info['error'] = "General loop: {}".format(err)
        return meta_info, []

    return meta_info, iter_colection


if __name__ == "__main__":
    print(" Start benchamrk ")

    # --- Benchmark config
    grad_uni = ModelsUnion(
        models=[GradientBoostingRegressor(n_estimators=200)],
        split_y=True)

    dummy_uni = ModelsUnion(
        models=[DummyRegressor(strategy="mean")], split_y=True)

    gp_sim = gp.GaussianProcessRegressor(
        kernel=KERNEL_SIMPLE, alpha=0, n_restarts_optimizer=10, normalize_y=True)

    gp_uni = ModelsUnion(models=[clone(gp_sim)], split_y=True)

    test_suites_small = [
        {
            'problem_name': ['zdt'],
            'portfolio': [[grad_uni, gp_sim, gp_uni]],
            'tuning_iter': [50],
            'prob_id': [1, 2, 3, 4, 5, 6],
            'prob_dim': [2, 5, 7, 12],
            'obj': [2],
        }
    ]

    #  config end
    #  ---

    i_total = 0

    meta_info = []
    loop_info = []

    for param_grid in test_suites_small:
        i = 0
        res = []
        for p in ParameterGrid(param_grid):
            i = i+1
            print("\n Evaluation.: {} \n i: {}".format(p, i))
            m_info, i_info = experiment_tutor(**p)
            meta_info.append(m_info)
            loop_info = loop_info + i_info


        i_total = i_total + i

        # File and path to folder
        prefix = ''.join(random.choices(
            string.ascii_lowercase + string.digits, k=10))
        rel_path_meta = '/benchmark_results/{}_tutor_meta_{}.csv'.format(
            param_grid['problem_name'][0], prefix)

        rel_path_loop = '/benchmark_results/{}_tutor_loop_{}.csv'.format(
            param_grid['problem_name'][0], prefix)

        path = os.path.dirname(os.path.abspath(__file__))

        # Write results
        print(" Total evaluations: {}".format(i_total))
        print(" Write meta. Path:{}".format(path + rel_path_meta))
        print(" Write loop. Path:{}".format(path + rel_path_loop))

        pd.DataFrame(meta_info).to_csv(path + rel_path_meta, mode='a+', index=False)
        pd.DataFrame(loop_info).to_csv(path + rel_path_loop, mode='a+', index=False)
