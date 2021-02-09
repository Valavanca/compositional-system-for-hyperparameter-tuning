
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

# -- Surrogates
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# --- Models
from src.composite._tutor_m import BaseTutor
from src.black_box import RF_experiment, psd_sample

import logging

logging.getLogger(__name__)

@pytest.fixture(scope="session")
def init_samples():
    logging.info(" Start initialoze samples \n\n")
    N_INIT = 6

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
    1-1
