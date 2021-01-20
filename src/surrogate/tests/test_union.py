import pytest
from sklearn.utils._testing import assert_raises
from sklearn import datasets
from sklearn.model_selection import train_test_split

# == estimators
from sklearn.gaussian_process import GaussianProcessRegressor as gp
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from sklearn.metrics import r2_score

from src.surrogate.surrogate_union import Union

kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))

def test_shape_single_predict():
    n_features = 10
    n_targets = 1

    X, y = datasets.make_regression(n_features=n_features, n_targets=n_targets)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    reg1 = GradientBoostingRegressor(random_state=1, n_estimators=10)
    reg2 = RandomForestRegressor(random_state=1, n_estimators=10)
    reg3 = LinearRegression()
    surrogates = [reg1, reg2, reg3]

    union_surr = Union(surrogates=surrogates, union_type="avarage")

    union_surr.fit(X_train, y_train)
    prediction = union_surr.predict(X_test)

    # 1D array
    assert prediction.ndim == 1
    # Prediction count
    assert prediction.shape[0] == len(X_test)

@pytest.mark.parametrize("union_type", ["avarage", "combination", "chain"])
def test_shape_multi_predict(union_type):
    n_features = 10
    n_targets = 3

    X, y = datasets.make_regression(n_features=n_features, n_targets=n_targets)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    reg1 = gp(random_state=11)
    reg2 = gp(random_state=22)
    reg3 = gp(random_state=33)
    surrogates = [reg1, reg2, reg3]

    union_surr = Union(surrogates=surrogates, union_type=union_type)

    union_surr.fit(X_train, y_train)
    prediction = union_surr.predict(X_test)

    # 2D array
    assert prediction.ndim == 2
    # Feature size
    assert prediction.shape[1] == n_targets
    # Prediction count
    assert prediction.shape[0] == len(X_test)

@pytest.mark.parametrize("union_type", ["avarage", "combination", "chain"])
def test_score_surr_union(union_type):

    n_features = 10
    n_targets = 3

    X, y = datasets.make_regression(n_features=n_features, n_targets=n_targets)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    reg1 = gp(random_state=11)
    reg2 = gp(random_state=22)
    reg3 = gp(random_state=33)
    surrogates = [reg1, reg2, reg3]

    union_surr = Union(surrogates=surrogates, union_type=union_type)

    union_surr.fit(X_train, y_train)
    score = union_surr.score(X_test, y_test, r2_score)

    assert score < 1
