from random import uniform
import logging
import pandas as pd
import numpy as np


logger = logging.getLogger()
logger.setLevel(logging.INFO)

class SamplesGenerator():
    """ Generate initial samples and update existing samples with new results.

    Args:
        problem (pagmo.problem): pygmo2 class
        X (array_like, shape (n_samples, n_features)): Samples features. Defaults to pd.DataFrame().
        y (array_like, shape (n_samples, )): Samples labels. Defaults to pd.DataFrame().
    """
    def __init__(self, problem, X=pd.DataFrame(), y=pd.DataFrame()):    
        self._problem = problem
        self._bounds = [d for d in zip(*problem.get_bounds())]
        self._nobj = problem.get_nobj()
        self._X = X
        self._y = y

    def random_sample(self, n=100):
        self._X = pd.DataFrame([[uniform(dim[0], dim[1]) for dim in self._bounds]
                                for _ in range(n)])
        self._X.columns = ['x{}'.format(i+1) for i in range(len(self._bounds))]

        self._y = self._X.apply(self._problem.fitness,
                                axis=1, result_type='expand')
        self._y.columns = ['f{}'.format(i+1) for i in range(self._nobj)]
        return self._X, self._y

    def return_X_y(self):
        if 0 not in (np.array(self._X).size, np.array(self._y).size):
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

        if self._X.empty and self._y.empty:
            logging.info("Initialization data generator")
            self._X = X_new
            self._y = y_new
        else:
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


        return self
