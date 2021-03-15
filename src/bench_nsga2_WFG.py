import warnings
import datetime
import time
import random
import string
import os
import logging
from pathlib import Path
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
from joblib import dump, load

import category_encoders as ce
import pygmo as pg

# Random forest as a black-box function


logging.basicConfig(filename='./bench_nsga2_WFG.log', level=logging.INFO)
# logging.getLogger().setLevel(logging.INFO)

warnings.filterwarnings('ignore')

def make_nd_pop(pro, x, y):
    nd_front = pg.fast_non_dominated_sorting(y)[0][0]
    nd_x = x[nd_front]
    nd_y = y[nd_front]
    t_pop = pg.population(pro)
    for i, p_vector in enumerate(nd_x):
        t_pop.push_back(x=p_vector, f=nd_y[i])
    return t_pop


def tuning_loop(problem_id,
                feature_dim,
                objectives,
                generation):
    logging.info(" --- Start loop \n\n")
    loop_start = time.time()
    iter_solution = []

    # --- WFG as black-box
    logging.info(f"\n WFG-{problem_id} initialization.{feature_dim} parameters and {objectives} objectives")
    udp = pg.wfg(prob_id=problem_id, dim_dvs=feature_dim,
                 dim_obj=objectives, dim_k=objectives-1)
    prob = pg.problem(udp)
    pop = pg.population(prob=prob, size=100)


    for i in range(generation):
        logging.info("\n generation --- {}".format(i))


        # --- save population
        temp = dict()
        temp['bench'] = 'nsga2'
        temp['problem'] = 'WFG'
        temp['dim'] = feature_dim
        temp['obj'] = objectives

        temp['pop_f'] = pop.get_f().tolist()
        temp['pop_size'] = len(pop.get_f())
        temp['iteration'] = i+1
        temp['time'] = time.time()
        temp['i_fevals'] = pop.problem.get_fevals()

        # --- rewrite population
        pop = pg.algorithm(pg.nsga2(gen = 1)).evolve(pop)

        nd_pop = make_nd_pop(prob, pop.get_x(), pop.get_f())
        temp["ndf_size"] = len(nd_pop.get_f())
        temp["ndf_f"] = nd_pop.get_f().tolist()
        temp["ndf_x"] = nd_pop.get_x().tolist()

        iter_solution.append(temp)

    loop = pd.DataFrame.from_dict(iter_solution)
    loop = loop.assign(prob_id=id(prob))

    # File and path to folder
    loop_prefix = ''.join(random.choices(
        string.ascii_lowercase + string.digits, k=6))

    rel_path = f'/bench/loop_NSGA2_WFG{problem_id}_{feature_dim}{objectives}.{loop_prefix}.pkl'
    path = os.path.dirname(os.path.abspath(__file__))

    # Write results
    print(" Write results. Path:{}".format(path + rel_path))
    logging.info(" Write results. Path:{}".format(path + rel_path))
    loop.to_pickle(path + rel_path)

if __name__ == "__main__":
    print(" === Start === ")

    test_set = [
        {
            'problem_id': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'feature_dim': [3],
            'objectives': [2],
            'generation': [500]
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
