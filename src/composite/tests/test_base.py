import pytest
from sklearn.utils._testing import assert_raises
from sklearn import datasets

# == estimators
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


from src.composite import BaseTutor


kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))

@pytest.mark.parametrize("final_estimator", [None, 'average', GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2)])
def test_smoke_fit(final_estimator):
    n_features = 100

    X, y = datasets.make_regression(n_features=n_features, n_targets=1)
    reg1 = GradientBoostingRegressor(random_state=1, n_estimators=10)
    reg2 = RandomForestRegressor(random_state=1, n_estimators=10)
    reg3 = LinearRegression()
    estimators = [reg1, reg2, reg3]

    bounds = ((-10,)*n_features, (10,)*n_features)
    tutor = BaseTutor(bounds, estimators, final_estimator)
    
    tutor.fit(X, y)
    tutor.predict()

def test_stacking():
    n_features = 10
    n = 5

    reg1 = GradientBoostingRegressor(random_state=1, n_estimators=10)
    reg2 = RandomForestRegressor(random_state=1, n_estimators=10)
    reg3 = LinearRegression()

    X, y = datasets.make_regression(n_features=n_features, n_targets=2)
    estimators = [reg1, reg3]
    bounds = ((-10,)*n_features, (10,)*n_features)
    tutor = BaseTutor(bounds, estimators)
    
    tutor.fit(X, y)
    predict = tutor.predict(n=n)
    # 2D array
    assert predict.ndim == 2
    # Feature size
    assert predict.shape[1] == n_features
    # Prediction count
    assert predict.shape[0] <= n
