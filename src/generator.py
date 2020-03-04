from random import uniform
import logging
import pandas as pd


logger = logging.getLogger()
logger.setLevel(logging.INFO)

class SamplesGenerator():

    def __init__(self, problem):
        self._problem = problem
        self._bounds = [d for d in zip(*problem.get_bounds())]
        self._nobj = problem.get_nobj()
        self._X = None
        self._y = None

    def random_sample(self, n=100):
        self._X = pd.DataFrame([[uniform(dim[0], dim[1]) for dim in self._bounds]
                                for _ in range(n)])
        self._X.columns = ['x{}'.format(i+1) for i in range(len(self._bounds))]

        self._y = self._X.apply(self._problem.fitness,
                                axis=1, result_type='expand')
        self._y.columns = ['f{}'.format(i+1) for i in range(self._nobj)]
        return self._X, self._y

    def return_X_y(self):
        if self._X is not None and self._y is not None:
            return self._X, self._y
        else:
            return pd.DataFrame(), pd.DataFrame()

    def update(self, X, y):
        if len(X) != len(y):
            raise Exception(
                'Broken data! Lengths X and y should be equal')
                
        X_new = pd.DataFrame(X)
        X_new.columns = ['x{}'.format(i+1) for i in range(len(self._bounds))]

        y_new = pd.DataFrame(y)
        y_new.columns = ['f{}'.format(i+1) for i in range(self._nobj)]

        if self._X is not None and self._y is not None:
            old_df = pd.concat([self._X, self._y], axis=1)
            upd_data = pd.concat([X_new, y_new], axis=1)

            dataset = pd.concat([old_df, upd_data]).drop_duplicates(
                subset=old_df.columns, keep='last')

            self._X = dataset[self._X.columns]
            self._y = dataset[self._y.columns]

            self._X.reset_index(drop=True, inplace=True)
            self._y.reset_index(drop=True, inplace=True)

            logging.info("In dataset add {} new results".format(
                len(self._X.index)-len(old_df.index)))
        else:
            logging.info("Initialization data generator")
            self._X = X_new
            self._y = y_new

        return self
