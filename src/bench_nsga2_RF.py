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


logging.basicConfig(filename='./bench_nsga2_RF.log', level=logging.INFO)
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

class RFProblem():

    def __init__(self, dataset_num):
        path = os.path.dirname(os.path.abspath(__file__))
        # --- describtion of random-forest(RF) parameters
        with open(path + '/randomforest-search-space.json', 'r') as File:
            params_description = json.loads(File.read())
        self._encoder = load(path + '/encoder.joblib')

        # DATA-SET
        # --- get evaluation dataset
        dataset = openml.datasets.get_dataset(dataset_num)
        X, y, cat, atr = dataset.get_data(
            dataset_format="array",
            target=dataset.default_target_attribute)

        # --- scale samples
        X = pd.DataFrame(X, columns=atr)
        ct = make_column_transformer(
            (StandardScaler(), make_column_selector(dtype_include=np.number)))
        X = pd.DataFrame(ct.fit_transform(X), columns=atr)

        self.X = X.values
        self.y = y

    def fitness(self, x):
        PARAMS = ['min_samples_split', 'min_samples_leaf', 'min_weight_fraction_leaf',
                  'criterion', 'max_depth', 'max_features', 'n_estimators']
        x_enc = self._encoder.inverse_transform(
            pd.DataFrame([x], columns=PARAMS))
        x_enc['n_estimators'] = x_enc['n_estimators'].astype('int32')
        param = x_enc.iloc[0].to_dict()     

        clf = RandomForestClassifier(**param)
        scores = cross_validate(clf,
                                self.X,
                                self.y,
                                scoring=["f1", "roc_auc"],
                                cv=3)

        for k, v in scores.items():
            scores[k] = np.mean(v)

        return [-1*scores['test_roc_auc'], scores['fit_time']]

    def get_nobj(self):
        return 2

    # Return bounds of decision variables
    def get_bounds(self):
        return ([0.0, 0.0, 0.0, 1.0, 2.0, 1.0, 10.0],
                [1.0e+00, 5.0e-01, 5.0e-01, 2.0e+00, 1.2e+02, 2.0e+00, 1.0e+03])

    # Integer Dimension
    def get_nix(self):
        return 4

    def gradient(self, x):
        return pg.estimate_gradient_h(lambda x: self.fitness(x), x)

    # Return function name
    def get_name(self):
        return "NSGA2 on RandomForestClassifier"


def tuning_loop(data_set, generation):
    logging.info(" --- Start loop \n\n")
    loop_start = time.time()
    iter_solution = []
    prob = pg.problem(RFProblem(data_set))
    pop = pg.population(prob=prob, size=100)


    for i in range(generation):
        logging.info("\n generation --- {}".format(i))
        # --- save population
        temp = dict()
        temp['objectives'] = ['test_roc_auc', 'fit_time']
        temp['pop_f'] = pop.get_f().tolist()
        temp['pop_size'] = len(pop.get_f())
        temp['i_gen'] = i+1
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

    rel_path = '/bench/{}_{}obj.{}.pkl'.format(
        prob.get_name(), prob.get_nobj(), loop_prefix)
    path = os.path.dirname(os.path.abspath(__file__))

    # Write results
    print(" Write meta. Path:{}".format(path + rel_path))
    logging.info(" Write meta. Path:{}".format(path + rel_path))
    # loop.to_csv(path + rel_path, mode='a+', index=False)
    loop.to_pickle(path + rel_path)

    # print(" Write dataset. Path:{}".format(path + '/bench/dataset.{}.pkl'.format(loop_prefix)))
    # dataset = pd.concat([params, score], axis=1)
    # dataset.to_pickle(path + '/bench/dataset.{}.pkl'.format(loop_prefix))




if __name__ == "__main__":
    print(" === Start === ")

    test_set = [
        {
            'data_set': [1049],
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
