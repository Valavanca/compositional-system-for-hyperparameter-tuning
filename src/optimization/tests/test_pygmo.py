import pytest
import random

from sklearn.utils._testing import assert_raises
from sklearn import datasets

from sklearn.gaussian_process import GaussianProcessRegressor as gp
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.utils._testing import assert_almost_equal


from src.optimization._pygmo_opt import PygmoProblem, Pygmo
from src.surrogate.surrogate_union import Union


def test_pred_from_custom_prob():
    n_features = 3
    n_check_points = 4
    n_targets = 2

    X, y = datasets.make_regression(n_targets=n_targets, n_features=n_features)

    # Instantiate a Gaussian Process model
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    reg = gp(kernel=kernel, n_restarts_optimizer=2)
    reg.fit(X, y)

    # Search bounds
    bounds = ((-10,)*n_features, (10,)*n_features)

    # Random points
    sample_test_X = []
    for _ in range(n_check_points):
        random_v = [random.randint(*b) for b in zip(bounds[0], bounds[1])]
        sample_test_X.append(random_v)

    # Direct prediction
    pred_etalon = reg.predict(sample_test_X)

    # Prediction from custom pygmo problem
    prob = PygmoProblem(reg, bounds)
    pred_test = [prob.fitness(v) for v in sample_test_X]

    assert_almost_equal(pred_test, pred_etalon)


def test_pygmo_singleobj():
    n_features = 10
    n_targets = 1

    X, y = datasets.make_regression(n_targets=n_targets, n_features=n_features)

    # Instantiate a Gaussian Process model
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    reg = gp(kernel=kernel, n_restarts_optimizer=2)
    reg.fit(X, y)

    # Search bounds
    bounds = ((-10,)*n_features, (10,)*n_features)

    opt = Pygmo(algo='simulated_annealing', pop_size=1,
                Ts=10., Tf=1e-5, n_T_adj=100)
    pred = opt.minimize(reg, bounds)

    # 0 - count of point, 1 - points dimensionality
    assert pred.shape[1] == n_features


@pytest.mark.parametrize("algo", ['nsga2', 'moead'])
def test_pygmo_multiobj(n_targets, algo):
    n_features = 10
    n_targets = 2

    X, y = datasets.make_regression(n_features=n_features, n_targets=n_targets,)

    # Instantiate a Gaussian Process model
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    reg = gp(kernel=kernel, n_restarts_optimizer=2)
    reg.fit(X, y)

    # Search bounds
    bounds = ((-10,)*n_features, (10,)*n_features)

    opt = Pygmo(algo, pop_size=100, gen=100)
    pred = opt.minimize(reg, bounds)

    assert pred.shape[1] == n_features  # 0 - count of point, 1 - points dimensionality

# --- combination: each regression corresponds to an appropriate objective
@pytest.mark.parametrize("union_type", ["combination"])
def test_pygmo_multiobj_union(union_type):
    n_features = 10
    n_targets = 3
    algo = 'nsga2'
    
    X, y = datasets.make_regression(n_features=n_features, n_targets=n_targets)

    # Instantiate a union of surrogates
    reg1 = gp(random_state=11)
    reg2 = gp(random_state=22)
    reg3 = gp(random_state=33)
    surrogates = [reg1, reg2, reg3]
    union_surr = Union(surrogates=surrogates, union_type=union_type)  
    union_surr.fit(X, y)

    # Search bounds
    bounds = ((-10,)*n_features, (10,)*n_features)

    opt = Pygmo(algo, pop_size=20, gen=100)
    pred = opt.minimize(union_surr, bounds)

    assert pred.shape[1] == n_features  # 0 - count of point, 1 - points dimensionality
