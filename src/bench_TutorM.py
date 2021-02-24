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

# Random forest as a black-box function
from black_box import RF_experiment, random_sample, psd_sample
from surrogate.surrogate_union import Union
from surrogate.custom_gp_kernel import KERNEL_MAUNA

logging.basicConfig(filename='./bench_TutorM_RF.log', level=logging.INFO)
# logging.getLogger().setLevel(logging.INFO)

warnings.filterwarnings('ignore')

from composite._tutor_m import TutorM

def tuning_loop(portfolio, 
                data_set: int,
                search_space: List[Mapping], 
                eval_budget: int, 
                solution_comb: str,
                cv_threshold: float,
                test_threshold: float,
                n_pred=1):

    N_INIT = 10

    # --- Data for random forest
    logging.info("\n Open evaluation dataset for random forest")
    dataset = openml.datasets.get_dataset(data_set)
    X, y, cat, atr = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute)

    X = pd.DataFrame(X, columns=atr)
    # y = pd.DataFrame(y, columns=['defect'])
    ct = make_column_transformer(
        (StandardScaler(), make_column_selector(dtype_include=np.number)))
    X = pd.DataFrame(ct.fit_transform(X), columns=atr)

    # --- Initial parameters for random forest 
    params = psd_sample(search_space, N_INIT, kind='sobol')
    score = pd.DataFrame([RF_experiment(X, y, config)
                          for idx, config in params.iterrows()])

    # --- Objective configurations
    # -1 : minimization
    # +1 : maximization
    obj_conf = {
        "fit_time": -1,
        "score_time": -1,
        "test_f1": 1,
        "test_roc_auc": 1
    }

    OBJECTIVES = ['test_roc_auc', 'fit_time']

    logging.info(" Start loop \n\n")
    loop_start = time.time()
    iter_solution = []
    i = 0

    n_iter = int(eval_budget/n_pred)
    while i < n_iter:
        i = i+1
        logging.info(f"\n iteration --- {i}")

        # --- [1] Build TutorM model and predict perspective solutions
        tutor = TutorM(portfolio,
                       cv_threshold=cv_threshold,
                       test_threshold=test_threshold
                       ).build_model(
            params, score, search_space)
        propos = tutor.predict(n=n_pred, kind=solution_comb)

        pred = dict()

        # --- [2] Evaluate proposed parameters
        propos_score = pd.DataFrame([RF_experiment(X, y, config)
                      for idx, config in propos.iterrows()])

        # --- update samples
        params = pd.concat([params, propos], ignore_index=True)
        score = pd.concat([score, propos_score], ignore_index=True)

        nd_front = pg.fast_non_dominated_sorting(
            (score[['test_roc_auc', 'fit_time']]*[-1, 1]).values)[0][0]
        nd_x = params.iloc[nd_front].values
        nd_f = score.iloc[nd_front][OBJECTIVES].values

        # hypervolume = pg.hypervolume(nd_f).compute(ref_point)
        # pred['hypervolume'] = hypervolume or None
        pred["ndf_size"] = len(nd_f)
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
        pred["samples_count"] = len(params)
        iter_solution.append(pred)

    # --- [3] collect results from the tuning-loop
    loop = pd.DataFrame(iter_solution)

    # File and path to folder
    loop_prefix = ''.join(random.choices(
        string.ascii_lowercase + string.digits, k=6))
    loop = loop.assign(loop_id=loop_prefix)

    rel_path = f'/bench/loop_TutorM.{loop_prefix}.pkl'
    path = os.path.dirname(os.path.abspath(__file__))

    # Write results
    print(" Write loop results. Path:{}".format(path + rel_path))
    # loop.to_csv(path + rel_path, mode='a+', index=False)
    loop.to_pickle(path + rel_path)

    print(" Write dataset. Path:{}".format(path + '/bench/dataset.{}.pkl'.format(loop_prefix)))
    dataset = pd.concat([params, score], axis=1)
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

    path = os.path.dirname(os.path.abspath(__file__))
    with open(path + '/randomforest-search-space.json', 'r') as File:
        search_space = json.loads(File.read())

    test_set = [
        {
            'portfolio': [[gp_mauna, grad, svr_rbf, mlp_reg]],
            'data_set': [1049],
            'search_space': [search_space],
            'eval_budget': [100],
            'solution_comb': ['stack'],
            'n_pred': [10],
            'cv_threshold': [0],
            'test_threshold': [0]
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
