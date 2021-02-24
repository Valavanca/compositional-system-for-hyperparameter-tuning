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


logging.basicConfig(filename='./bench_category_RF.log', level=logging.INFO)
# logging.getLogger().setLevel(logging.INFO)

warnings.filterwarnings('ignore')


class PygmoProblem():

    def __init__(self,
                 surrogate,
                 bounds, integer):
        self._surrogate = surrogate
        self._bounds = bounds
        self._int = integer

    def fitness(self, x):
        x = np.array(x)
        result = self._surrogate.predict(x.reshape(1, -1))
        return result.flatten().tolist()

    def get_nobj(self):
        prediction = self._surrogate.predict([self._bounds[0]])
        nobj = prediction.flatten().size
        return nobj

    # Return bounds of decision variables
    def get_bounds(self):
        return self._bounds

    # Integer Dimension
    def get_nix(self):
        return self._int

    def gradient(self, x):
        return pg.estimate_gradient_h(lambda x: self.fitness(x), x)

    # Return function name
    def get_name(self):
        meta_name = type(self._surrogate).__name__
        if hasattr(self._surrogate, 'surrogates'):
            return self._surrogate._union_type + "(" + "+".join([type(t).__name__ for t in self._surrogate.surrogates]) + ")"
        else:
            return meta_name


def get_bounds(desc, params_enc):
    bounds = []
    kv_desc = {p['name']: p for p in desc}
    for col in params_enc.columns:
        if 'categories' in kv_desc[col]:
            bounds.append(list(range(1, len(kv_desc[col]['categories'])+1)))
        else:
            bounds.append(kv_desc[col]['bounds'])

    return list(zip(*bounds))


def tuning_loop(surrogate, 
                data_set, 
                search_space: List[Mapping], 
                eval_budget: int, 
                union_type='single', 
                n_pred=1):
    N_INIT = 10
    OBJECTIVES = ['test_roc_auc', 'fit_time']

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

    logging.info(" Start loop \n\n")
    loop_start = time.time()
    iter_solution = []
    i = 0

    n_iter = int(eval_budget/n_pred)
    while i < n_iter:
        i = i+1
        logging.info("\n iteration --- {}".format(i))

        # --- [1] encode categorical RF-parameters
        cat_columns = params.select_dtypes(include=[object]).columns
        encoder = ce.OrdinalEncoder(cols=cat_columns)

        params_enc = encoder.fit_transform(params)
        params_enc.sort_index(axis=1, 
                            ascending=False,
                            key=lambda c: params_enc[c].dtypes, 
                            inplace=True)

        null_count = score.isnull().sum().sum()
        if null_count > 0:
            logging.warning(
                " There are {} empty values in the sample table ".format(null_count))
        
        obj_sign_vector = []
        for c in score.columns:
            if c in obj_conf.keys():
                obj_sign_vector.append(obj_conf[c])
            else:
                obj_sign_vector.append(-1)
                logging.warning(
                    "There is no configuration for objective {}".format(c))

        obj = (score * -1 * obj_sign_vector)[OBJECTIVES]  # inverse. maximization * -1 == minimization

        # --- [2] fit surrogate model with RF-parameters
        if union_type == "separate":
            model = Union(surrogates=[surrogate]*len(OBJECTIVES), union_type="separate")
        elif union_type == "chain":
            model = Union(surrogates=[surrogate], union_type="chain")
        elif union_type == "avarage":
            model = Union(surrogates=surrogate, union_type="avarage")
        elif union_type == "single":
            model = surrogate
        model.fit(params_enc.values, obj.values)

        # --- [3] create optimization problem that based on surrogate model
        bounds = get_bounds(search_space, params_enc)
        integer_params = (params_enc.dtypes == np.int64).sum()
        prob = pg.problem(PygmoProblem(model, bounds, integer_params))

        # --- [4] optimization
        isl = pg.island(algo=pg.nsga2(gen=300), # NSGA2
                size=100,
                prob=prob,
                udi=pg.mp_island(),
                )
        isl.evolve()
        isl.wait()
        final_pop = isl.get_population()

        idx = pg.fast_non_dominated_sorting(final_pop.get_f())[0][0]
        pop_df = pd.DataFrame(final_pop.get_x()[idx], columns=params_enc.columns)
        pop_df['n_estimators'] = pop_df['n_estimators'].astype('int32')
        propos = encoder.inverse_transform(pop_df)

        # random select the required amount of predictions
        n = n_pred if len(propos) > n_pred else len(propos)
        logging.info(
            " Select {} from {} proposed configurations".format(n, len(propos)))
        propos = propos.sample(n=n)

        pred = dict()
        pred['iteration'] = i
        pred['problem'] = prob.get_name()
        pred['objectives'] = OBJECTIVES
        pred['feature_dim'] = prob.get_nx()

        # --- [5] Evaluate proposed parameters
        propos_score = pd.DataFrame([RF_experiment(X, y, config)
                      for idx, config in propos.iterrows()])

        # --- update samples
        params = pd.concat([params, propos], ignore_index=True)
        score = pd.concat([score, propos_score], ignore_index=True)

        # ----------------------                                                             Hypervolume
        #  hypervolume is complicated to calculate when reference point is not stable
        # ref_point = pg.nadir(obj.to_numpy())
        # pred['ref_point'] = ref_point
        nd_front = pg.fast_non_dominated_sorting(
            (score[['test_roc_auc', 'fit_time']]*[-1, 1]).values)[0][0]
        nd_x = params.iloc[nd_front].values
        nd_f = score.iloc[nd_front][OBJECTIVES].values

        # hypervolume = pg.hypervolume(nd_f).compute(ref_point)
        # pred['hypervolume'] = hypervolume or None
        pred["ndf_size"] = len(nd_f)
        pred["ndf_f"] = nd_f.tolist()
        pred["ndf_x"] = nd_x.tolist()

        pred["i_time"] = time.time() - loop_start
        iter_solution.append(pred)


    loop = pd.DataFrame(iter_solution)
    loop = loop.assign(model_id=id(prob))

    # File and path to folder
    loop_prefix = ''.join(random.choices(
        string.ascii_lowercase + string.digits, k=6))
    rel_path = '/bench/{}_{}_RF.{}.pkl'.format(
        prob.get_name(), prob.get_nobj(), loop_prefix)
    path = os.path.dirname(os.path.abspath(__file__))

    # Write results
    print(" Write meta. Path:{}".format(path + rel_path))
    # loop.to_csv(path + rel_path, mode='a+', index=False)
    loop.to_pickle(path + rel_path)

    print(" Write dataset. Path:{}".format(path + '/bench/dataset.{}.pkl'.format(loop_prefix)))
    dataset = pd.concat([params, score], axis=1)
    dataset.to_pickle(path + '/bench/dataset.{}.pkl'.format(loop_prefix))




if __name__ == "__main__":
    print(" <...> ")

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

    with open('src/randomforest-search-space.json', 'r') as File:
        search_space = json.loads(File.read())

    test_set = [
        {
            'surrogate': [gp_mauna],
            'data_set': [1049],
            'search_space': [search_space],
            'eval_budget': [1000],
            'union_type': ['single'],
            'n_pred': [10]
        },
        {
            'surrogate': [grad, svr_rbf, mlp_reg],
            'data_set': [1049],
            'search_space': [search_space],
            'eval_budget': [1000],
            'union_type': ['separate'],
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
