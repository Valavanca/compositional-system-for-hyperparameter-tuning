from src.composite import TutorM
from sklearn.utils._testing import assert_raises

# == estimators
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

def test_init_tutorm():
    reg1 = GradientBoostingRegressor(random_state=1, n_estimators=10)
    reg2 = RandomForestRegressor(random_state=1, n_estimators=10)
    reg3 = LinearRegression()
    estimators = [('gb', reg1), ('rf', reg2), ('lr', reg3)]
    # estimators = [reg1, reg2, reg3]
    bounds = ((0, 0), (5, 5))

    TutorM(bounds, estimators)
