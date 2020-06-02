import pytest
import numpy as np


from sklearn import datasets
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.utils._testing import assert_almost_equal

from src.composite._models_union import ModelsUnion


def test_single_obj_regression():
    X, y = datasets.make_regression(n_targets=1)
    X_train, y_train = X[:50], y[:50]
    X_test, y_test = X[50:], y[50:]

    rgr = GradientBoostingRegressor(random_state=0)
    rgr.fit(X_train, y_train)
    references = rgr.predict(X_test)

    rgr = ModelsUnion([GradientBoostingRegressor(random_state=0)])
    rgr.fit(X_train, y_train)
    y_pred = rgr.predict(X_test)

    assert_almost_equal(references, y_pred)



def test_multi_obj_regression(): 
    X, y = datasets.make_regression(n_targets=3)
    X_train, y_train = X[:50], y[:50]
    X_test, y_test = X[50:], y[50:]

    references = np.zeros_like(y_test)
    for n in range(3):
        rgr = GradientBoostingRegressor(random_state=0) 
        rgr.fit(X_train, y_train[:, n])
        references[:, n] = rgr.predict(X_test)

    rgr = ModelsUnion([GradientBoostingRegressor(random_state=0)], split_y=True)
    rgr.fit(X_train, y_train)
    y_pred = rgr.predict(X_test) 

    assert_almost_equal(references, y_pred)

def test_mean_multi_obj_regression():
    X, y = datasets.make_regression(n_targets=3)
    X_train, y_train = X[:50], y[:50]
    X_test, y_test = X[50:], y[50:]

    predictions = []
    for _ in range(3):
        rgr = MLPRegressor(random_state=0)
        rgr.fit(X_train, y_train)
        predictions.append(rgr.predict(X_test))
    references = np.array(predictions).mean(axis=0)

    rgr = ModelsUnion([MLPRegressor(random_state=0),
                       MLPRegressor(random_state=0), 
                       MLPRegressor(random_state=0)])
    rgr.fit(X_train, y_train)
    y_pred = rgr.predict(X_test)

    assert_almost_equal(references, y_pred)
