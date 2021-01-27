
import json
from collections import OrderedDict

import numpy as np
import pytest
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification, make_regression
# -- Surrogates
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# --- Models
from model.tree_parzen_estimator import TreeParzenEstimator
from model._tutor_m import BaseTutor

# --- Internal

from core_entities.search_space import (Hyperparameter, get_dimensionality,
                                        initialize_hyperparameter)
from selection.mersenne_twister import MersenneTwister
from model._optimization._pygmo_opt import Pygmo


@pytest.fixture(scope="session")
def init_samples():
    # --- Description for search space and selector
    n_samples = 7
    searc_space_path = 'main_node/model/tests/OpenmlRFv2.json'
    searc_space_description = read_file(searc_space_path)
    selector = MersenneTwister()

    # --- Initialization of search space
    configs, sp = init_search_space(searc_space_description, selector, n_samples)
    score = [RF_sklearn(**conf) for conf in configs]

    return {"search_space": sp,
            "search_space_json": searc_space_description,
            "configs": configs,
            "configs_score": score}


def test_base_tutor_RF(init_samples):

    searc_space_json = init_samples['search_space_json']

    X = pd.DataFrame(init_samples['configs'])
    y = pd.DataFrame(init_samples['configs_score'])

    # --- surrogates
    reg1 = GradientBoostingRegressor(random_state=1, n_estimators=10)
    reg2 = RandomForestRegressor(random_state=1, n_estimators=10)

    # --- optimezers
    opt1 = Pygmo('nsga2', pop_size=100, gen=100)
    opt2 = Pygmo('moead', pop_size=100, gen=100)

    # --- predictor: tutor
    tutor = BaseTutor([reg2], [opt1, opt2])
    tutor.build_model(X, y, None, True, sp_json=searc_space_json)
    n_pred = 10
    pred = tutor.predict(n_pred)


def test_base_tutor_predict():
    n_features = 10
    X, y = make_regression(n_samples=1000,
                           n_features=n_features, 
                           n_targets=2, 
                           noise=20, 
                           random_state=0)

    # --- surrogates
    reg1 = GradientBoostingRegressor(random_state=1, n_estimators=10)
    reg2 = RandomForestRegressor(random_state=1, n_estimators=10)

    # --- optimezers
    opt1 = Pygmo('nsga2', pop_size=100, gen=100)
    opt2 = Pygmo('moead', pop_size=100, gen=100)

    # --- predictor: tutor
    tutor = BaseTutor([reg1, reg2], [opt1, opt2])
    tutor.build_model(X, y, None, True)
    n_pred = 10
    pred = tutor.predict(n_pred)

    assert pred.shape[1] == n_features  # 0 - count of point, 1 - points dimensionality
    assert pred.ndim == 2  # 2D array
    assert pred.shape[0] <= n_pred  # prediction count
  

def test_TPE(init_samples):
    parameters = {
        "top_n_percent": 30,
        "random_fraction": 0,
        "bandwidth_factor": 3,
        "min_bandwidth": 0.001,
        "SamplingSize": 96
    }
    model = TreeParzenEstimator(parameters)

    X = pd.DataFrame(init_samples['configs'])
    y = pd.DataFrame(init_samples['configs_score'])

    # ! remove 'sample' after it will be clear how to create 'search space'
    sample = init_samples['configs'][0]
    sample["experiment"] = "RF"
    description = init_samples['search_space'].describe(sample)
    is_minimization = False

    model.build_model(features=X, labels=y, features_description=description, is_minimization=is_minimization)
    model.predict()

# =============================== Utilities =====================================
def init_search_space(desc: str, selector, n=1):
    # --- search space
    dimensionality = get_dimensionality(desc, 0) - 1
    sp = initialize_hyperparameter(desc, selector, dimensionality)

    # --- random configurations
    conf = []
    for _ in range(n):
        rand_conf = OrderedDict()
        for _ in range(dimensionality):
            sp.generate(rand_conf)
        del rand_conf["experiment"]
        conf.append(rand_conf)

    return conf, sp

def read_file(path_to_file: str):
    with open(path_to_file, 'r') as File:
        jsonFile = json.loads(File.read())
        return jsonFile


def RF_sklearn(**RF_kwargs):

    # import a dataset from openml
    # Dataset_ID = 1049
    # dataset = openml.datasets.get_dataset(Dataset_ID)
    # X, y, _, _ = dataset.get_data(dataset_format="array", target=dataset.default_target_attribute)

    # linearly separable
    X, y = make_classification(n_samples=1000,
                               n_features=20,
                               n_informative=15,
                               n_redundant=2,
                               n_repeated=2,
                               n_classes=2,
                               shuffle=False,
                               random_state=0)

    # Build sklearn RF pipeline
    clf = make_pipeline(StandardScaler(), RandomForestClassifier(**RF_kwargs))

    # evaluate classifier with cross_validation
    scores = cross_validate(clf, X, y,
                            scoring=["f1_macro", "f1_weighted", "roc_auc"],
                            cv=3,
                            n_jobs=-1)

    # mean scores for all folds
    for k, v in scores.items():
        scores[k] = np.mean(v)

    return scores
