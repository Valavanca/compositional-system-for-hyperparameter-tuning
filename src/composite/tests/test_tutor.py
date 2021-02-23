
import json
import logging
from os.path import abspath
from collections import OrderedDict
import openml

import numpy as np
import pytest
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification, make_regression
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

# -- Surrogates
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# --- Models
from src.composite._tutor_m import BaseTutor, TutorM
from src.black_box import RF_experiment, psd_sample, random_sample

import logging

logging.getLogger(__name__)

@pytest.fixture(scope="session")
def init_samples():
    logging.info(" start initialize samples \n\n")
    N_INIT = 12

    # --- Parameters describtion
    with open('src/randomforest-search-space.json', 'r') as File:
        params_description = json.loads(File.read())

    # --- Data for random forest
    logging.info("\n Open evaluation dataset for random forest")
    dataset = openml.datasets.get_dataset(1049)
    X, y, cat, atr = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute)

    X = pd.DataFrame(X, columns=atr)
    # y = pd.DataFrame(y, columns=['defect'])
    ct = make_column_transformer(
        (StandardScaler(), make_column_selector(dtype_include=np.number)))
    X = pd.DataFrame(ct.fit_transform(X), columns=atr)

    # --- Evaluate initial parameters for the random forest
    params = psd_sample(params_description, N_INIT, kind='sobol')
    score = pd.DataFrame([RF_experiment(X.values, y, config)
                          for idx, config in params.iterrows()])

    return {"params_desc": params_description,
            "params": params,
            "params_score": score}


def test_base_tutor_RF(init_samples):
    logging.info(" test BaseTutor on Random Forest \n")

    # --- Iniial data and configuration
    params_desc = init_samples['params_desc']
    params = pd.DataFrame(init_samples['params'])
    score = pd.DataFrame(init_samples['params_score'])

    # --- surrogates
    reg1 = GradientBoostingRegressor(random_state=1, n_estimators=10)
    reg2 = RandomForestRegressor(random_state=2, n_estimators=10)

    # --- predictor
    tutor = BaseTutor([reg1, reg2], union_type='separate')
    tutor.build_model(params, score, params_desc)
    pred = tutor.predict(10)


def test_score_tutor_RF(init_samples):
    logging.info(" test score from BaseTutor on Random Forest \n")

    # --- Iniial data and configuration
    params_desc = init_samples['params_desc']
    params = pd.DataFrame(init_samples['params'])
    score = pd.DataFrame(init_samples['params_score'])

    # --- surrogates
    reg1 = GradientBoostingRegressor(random_state=1, n_estimators=10)
    reg2 = RandomForestRegressor(random_state=2, n_estimators=10)

    # --- predictor
    tutor = BaseTutor([reg1, reg2], union_type='separate')
    tutor.build_model(params, score, params_desc)

    # --- Test parameters
    dataset = openml.datasets.get_dataset(1049)
    X, y, cat, atr = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute)

    X = pd.DataFrame(X, columns=atr)
    ct = make_column_transformer(
        (StandardScaler(), make_column_selector(dtype_include=np.number)))
    X = pd.DataFrame(ct.fit_transform(X), columns=atr)
    params_test = random_sample(params_desc, 10)
    obj_test = pd.DataFrame([RF_experiment(X.values, y, config)
                               for idx, config in params_test.iterrows()])
    
    score = tutor.score(params_test, obj_test, r2_score)
    
    assert len(score) == 2 # 2 objectives


def test_tutor_m_build(init_samples):

    # --- Iniial data and configuration
    params_desc = init_samples['params_desc']
    params = pd.DataFrame(init_samples['params'])
    obj = pd.DataFrame(init_samples['params_score'])

    portfolio = [
        GaussianProcessRegressor(n_restarts_optimizer=20),
        GradientBoostingRegressor(n_estimators=500),
        SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1),
        MLPRegressor(hidden_layer_sizes=(20, 60, 20),
                     activation='relu', solver='lbfgs')
    ]

    TutorM(portfolio).build_model(params, obj, params_desc).predict()
