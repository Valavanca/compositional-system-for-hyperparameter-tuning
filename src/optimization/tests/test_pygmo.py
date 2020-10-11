import pytest
import random

from sklearn.utils._testing import assert_raises
from sklearn import datasets

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.utils._testing import assert_almost_equal


from src.optimization import Pygmo, CustomProblem


def test_pred_custom_prob():
    n_features = 3
    n_check_points = 4
    n_targets = 2

    X, y = datasets.make_regression(n_targets=n_targets, n_features=n_features)

    # Instantiate a Gaussian Process model
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    reg = GaussianProcessRegressor(kernel=kernel,
                                   n_restarts_optimizer=2)
    reg.fit(X, y)

    # Search bounds
    bounds = ((-10,)*n_features, (10,)*n_features)

    # Random points points
    samples_v = []
    for _ in range(n_check_points):
        random_v = [random.randint(*b) for b in zip(bounds[0], bounds[1])]
        samples_v.append(random_v)

    # Direct prediction
    pred_etalon = reg.predict(samples_v)

    # Prediction from custom pygmo problem
    prob = CustomProblem(reg, bounds)
    pred_test = [prob.fitness(v) for v in samples_v]

    assert_almost_equal(pred_test, pred_etalon)


@pytest.mark.parametrize("n_targets", [2, 5])
@pytest.mark.parametrize("algo", ['nsga2'])
def test_pygmo_multiobj(n_targets, algo):
    n_features = 10
    X, y = datasets.make_regression(n_targets=n_targets, n_features=n_features)

    # Instantiate a Gaussian Process model
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    reg = GaussianProcessRegressor(kernel=kernel,
                                   n_restarts_optimizer=2)
    reg.fit(X, y)

    # Search bounds
    bounds = ((-10,)*n_features, (10,)*n_features)
    opt = Pygmo(reg, bounds, algo=algo, gen=100, pop_size=100)

    pred = opt.predict()
    assert pred.shape[1] == n_features # 0 - count of point, 1 - points dimensionality


def test_pygmo_singleobj():
    n_targets = 1
    n_features = 10
    X, y = datasets.make_regression(n_targets=n_targets, n_features=n_features)

    # Instantiate a Gaussian Process model
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    reg = GaussianProcessRegressor(kernel=kernel,
                                   n_restarts_optimizer=2)
    reg.fit(X, y)

    # Search bounds
    bounds = ((-10,)*n_features, (10,)*n_features)
    opt = Pygmo(reg, bounds, algo='simulated_annealing', pop_size=1,
                Ts=10., Tf=1e-5, n_T_adj=100)

    pred = opt.predict()
    # 0 - count of point, 1 - points dimensionality
    assert pred.shape[1] == n_features
