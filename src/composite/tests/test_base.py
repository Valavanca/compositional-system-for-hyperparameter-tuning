from src.composite import BaseTutor
from sklearn.utils._testing import assert_raises
from sklearn import datasets

# == estimators
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression


def test_simple_stack():
    X, y = datasets.make_regression(n_targets=1)
    X_train, y_train = X[:50], y[:50]
    X_test, y_test = X[50:], y[50:]


    reg1 = GradientBoostingRegressor(random_state=1, n_estimators=10)
    reg2 = RandomForestRegressor(random_state=1, n_estimators=10)
    reg3 = LinearRegression()
    estimators = [reg1, reg2, reg3]
    bounds = ((0, 0), (5, 5))
    tutor = BaseTutor(bounds, estimators)
    
    tutor.fit(X,y)
