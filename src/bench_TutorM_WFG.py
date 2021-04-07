import warnings
import datetime
import time
import random
import string
import os
import logging
import json
from typing import List, Mapping

import openml
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pprint import pformat

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import clone
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor


import category_encoders as ce
import pygmo as pg
import sobol_seq


# Random forest as a black-box function
from black_box import RF_experiment, random_sample, psd_sample
from surrogate.surrogate_union import Union
from surrogate.custom_gp_kernel import KERNEL_MAUNA

logging.basicConfig(filename='./bench_TutorM_WFG.log', level=logging.DEBUG)
# logging.getLogger().setLevel(logging.INFO)

warnings.filterwarnings('ignore')

from composite._tutor_m_PYGMO import TutorM


def psd_sample(bounds,  n=1, kind='sobol'):
    """ Pseudo-random configuration sampling

    Args:
        n (int, optional): number of configurations to generate. Defaults to 1.
        kind (str, optional): identifying the general kind of pseudo-random samples generator. Sobol('sobol') or Latin Hypercube sampling('lh'). Defaults to 'sobol'.

    Raises:
        ValueError: wrong value for pseudo-samples generation

    Returns:
        pandas.DataFrame: configurations samples
    """

    dim = len(bounds[0])  # bounds in pygmo format

    if kind == 'sobol':
        squere = sobol_seq.i4_sobol_generate(dim, n)
    elif kind == 'lh':
        squere = np.random.uniform(size=[n, dim])
        for i in range(0, dim):
            squere[:, i] = (np.argsort(squere[:, i])+0.5)/n
    else:
        raise ValueError(
            "`kind` property could be `sobol` or `lh`. Received: ", kind)

    conf = (bounds[1]-bounds[0])*squere + bounds[0]
    return pd.DataFrame(conf).add_prefix('x_')

def tuning_loop(portfolio, 
                problem_id: int,
                objectives: int,
                feature_dim: int,
                eval_budget: int, 
                n_init: int,
                train_test_sp: float,
                cv_threshold: float,
                test_threshold: float,
                solution_comb: str,
                n_pred: int):

    # --- WFG as black-box
    logging.info(f"\n WFG-{problem_id} initialization.{feature_dim} parameters and {objectives} objectives")
    udp = pg.wfg(prob_id=problem_id, dim_dvs=feature_dim,
                 dim_obj=objectives, dim_k=objectives-1)
    prob = pg.problem(udp)

    # --- Initial parameters for random forest 
    bounds = prob.get_bounds()
    df_params = psd_sample(bounds, n_init, kind='sobol')
    df_obj = df_params.apply(prob.fitness, axis=1,
                             result_type='expand').add_prefix('f_')

    logging.info(" Start loop \n\n")
    loop_start = time.time()
    iter_solution = []
    i = 0

    # --- Stats from initial samples
    pred = dict()
    pred['final_surr'] = 'sobol'
    pred['bench'] = 'portfolio'
    pred['problem'] = 'WFG'
    pred['dim'] = feature_dim
    pred['obj'] = objectives
    pred['init'] = n_init
    pred['iteration'] = i

    nd_front = pg.fast_non_dominated_sorting(df_obj.values)[0][0]
    nd_x = df_params.iloc[nd_front].values
    nd_f = df_obj.iloc[nd_front].values

    pred["ndf_size"] = len(nd_front)
    pred["ndf_f"] = nd_f.tolist()
    pred["ndf_x"] = nd_x.tolist()

    pred["solution_pool"] = 0
    pred["i_time"] = time.time() - loop_start
    pred["surr_id"] = None
    pred["samples_count"] = len(df_params)

    iter_solution.append(pred)

    n_iter = int(eval_budget/n_pred)
    while i < n_iter:
        pred = dict()
        i = i+1
        pred['problem'] = 'WFG'
        pred['dim'] = feature_dim
        pred['obj'] = objectives
        pred['init'] = n_init
        pred['iteration'] = i
        pred['n_pred'] = n_pred
        pred['bench'] = 'portfolio'
        logging.info(f"\n\n iteration --- {i}")

        # --- [1] Build TutorM model and predict perspective solutions
        # ! check the version of TutorM
        bounds = prob.get_bounds()
        tutor = TutorM(portfolio,
                       train_test_sp=train_test_sp,
                       cv_threshold=cv_threshold,
                       test_threshold=test_threshold
                       ).build_model(
            df_params, df_obj, bounds)
        pred_params = tutor.predict(n=n_pred, kind=solution_comb)

        

        # --- [2] Evaluate proposed parameters
        propos_obj = pd.DataFrame([prob.fitness(x).tolist()
                                   for idx, x in pred_params.iterrows()]).add_prefix('f_')

        # --- update samples
        df_params = pd.concat([df_params, pred_params], ignore_index=True)
        df_obj = pd.concat([df_obj, propos_obj], ignore_index=True)

        # --- get stats from updated samples
        nd_front = pg.fast_non_dominated_sorting(df_obj.values)[0][0]
        nd_x = df_params.iloc[nd_front].values
        nd_f = df_obj.iloc[nd_front].values

        pred["ndf_size"] = len(nd_front)
        pred["ndf_f"] = nd_f.tolist()
        pred["ndf_x"] = nd_x.tolist()

        if tutor._status['final_surr']:
            pred["final_surr"] = [[type(surr).__name__ for surr in tut._surrogates]
                            for tut in tutor._status['final_surr']]
        else:
            pred["final_surr"] = ['sobol']

        pred["solution_pool"] = len(tutor._status['solutions'])

        pred["i_time"] = time.time() - loop_start
        pred["tutor_id"] = id(tutor)
        pred["samples_count"] = len(df_params)


        iter_solution.append(pred)

    # --- [3] collect results from the tuning-loop
    loop = pd.DataFrame(iter_solution)

    # File and path to folder
    loop_prefix = ''.join(random.choices(
        string.ascii_lowercase + string.digits, k=6))
    loop = loop.assign(loop_id=loop_prefix)

    rel_path = f'/bench/loop_TutorM_WFG{problem_id}_{feature_dim}{objectives}.{loop_prefix}.pkl'
    path = os.path.dirname(os.path.abspath(__file__))

    # Write results
    print(" Write loop results. Path:{}".format(path + rel_path))
    # loop.to_csv(path + rel_path, mode='a+', index=False)
    loop.to_pickle(path + rel_path)

    print(" Write dataset. Path:{}".format(path + '/bench/dataset.{}.pkl'.format(loop_prefix)))
    dataset = pd.concat([df_params, df_obj], axis=1)
    dataset.to_pickle(path + '/bench/dataset.{}.pkl'.format(loop_prefix))


if __name__ == "__main__":
    print(" === Start === ")

    # --- 1
    gp_mauna = GaussianProcessRegressor(
        kernel=KERNEL_MAUNA, n_restarts_optimizer=20)

    # --- 2
    grad = GradientBoostingRegressor(n_estimators=500)

    # --- 3
    svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)

    # --- 4
    mlp_reg = MLPRegressor(hidden_layer_sizes=(
        20, 60, 20), activation='relu', solver='lbfgs')

    test_set = [
        {
            'portfolio': [[gp_mauna, grad, svr_rbf, mlp_reg]],
            'problem_id': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'objectives': [2],
            'feature_dim': [3],
            'eval_budget': [900],
            'n_init': [100],
            'train_test_sp': [0.25],
            'cv_threshold': [-10000],
            'test_threshold': [-10000],
            'solution_comb': ['stack'],
            'n_pred': [10]
        }
    ]

    logging.info(pformat(test_set))

    i_total = 0
    with Parallel(prefer='threads') as parallel:
        for param_grid in test_set:
            grid = ParameterGrid(param_grid)
            total_comb = len(grid)
            logging.info(
                "\n\n Total combinations in round: {}".format(total_comb))

            parallel(delayed(tuning_loop)(**p) for p in grid)

    print("\n === Finish === ")
