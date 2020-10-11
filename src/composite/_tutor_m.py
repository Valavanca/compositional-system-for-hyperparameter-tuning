from typing import List, Tuple
import hashlib
import logging
from random import uniform
from collections import defaultdict

import pygmo as pg
import pandas as pd
import sobol_seq
import numpy as np

# -- utilities
from sklearn import clone
from sklearn.model_selection import cross_validate
from sklearn.base import TransformerMixin, BaseEstimator, MetaEstimatorMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted, check_X_y
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed

# -- ensemble
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble._stacking import _BaseStacking
from sklearn.ensemble import GradientBoostingRegressor
from src.composite import ModelsUnion

# -- estimators
from sklearn.linear_model import LinearRegression
import sklearn.gaussian_process as gp
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel, RationalQuadratic, ExpSineSquared


# from .search import Gaco, Nsga2
# from search import Gaco, Nsga2, RandS, MOEActr

from src.optimization import Pygmo

logger = logging.getLogger()
logger.setLevel(logging.INFO)
# np.random.seed(1)


class BaseTutor(RegressorMixin, _BaseStacking):
    """ fit estimators and search perspective points with optimization algorithms

    1. Fit and Stack estimators. Assume that each estimator could represent a complete search space.
    2. Apply optimization algorithm(s)
    3. Return a found solution

    Reference:
        - sklearn.ensemble.StackingRegressor: Stacked generalization consists in stacking the output of individual estimator and use a regressor to compute the final prediction.
        - sklearn.ensemble.VotingRegressor: A voting regressor is an ensemble meta-estimator that fits several base regressors, each on the whole dataset. Then it averages the individual predictions to form a final prediction.
    """    
    def __init__(self, bounds, estimators, final_estimator=None, verbose=0):
        """
        Args:
            bounds (tuple of tuples): parameters bounds. Example: ((0, 0), (5, 5))
            estimators (list of sklearn.estimators): [description]
            final_estimator (str or sklearn.estimator, optional): Merger predictionі from estimators. None, "average" or estiamtor. Defaults to None.
            verbose (int, optional): Verbosity level. Defaults to 0.
        """        
        super().__init__(estimators=_name_estimators(estimators))
        self.final_estimator = final_estimator
        self.bounds = bounds
        self.verbose = verbose
        self.estimators_ = None
        self.optimizators_ = []

    def fit(self, X, y):
        """ fit and stack selected estimators
        """
        X, y = check_X_y(X, y,
                         multi_output=True,
                         accept_sparse=True,
                         estimator=self,
                         )

        # --- Fit estimators
        if isinstance(self.final_estimator, BaseEstimator):
            # --- Variant 1: Stack estimators with the final estimator
            reg = StackingRegressor(
                estimators=self.estimators,
                final_estimator=self.final_estimator,
                verbose=self.verbose)
            reg.fit(X, y)
            self.estimators_ = [reg]
        elif self.final_estimator == "average":
            # --- Variant 2: Uniformly average all predictions by VotingRegressor
            reg = VotingRegressor(estimators=self.estimators)
            reg.fit(X, y)
            self.estimators_ = [reg]
        else:
            # --- Variant 3: Drop VotingRegressor and use trained sub-estimators
            reg = VotingRegressor(estimators=self.estimators)
            reg.fit(X, y)
            self.estimators_ = reg.estimators_

        self._opt_search()
        return self

    def _opt_search(self, **params):
        """ Apply optimization algorithm(s)
        """ 
        self.optimizators_ = []       
        for est in self.estimators_:
          check_is_fitted(est)
          opt = Pygmo(est, self.bounds, algo='nsga2', gen=100, pop_size=100)
          self.optimizators_.append(opt)

    
    def predict(self, X=None, y=None, n=1, **predict_params):
        return [opt.predict() for opt in self.optimizators_]
        




class TutorM(RegressorMixin, _BaseStacking):
    """ This class combines the estimators' hypothesis about parameter and objective space with an optimization technique to find perspective or optimal points in objective space.

        The single parameter combination is interpreted as a point in parameter space. This space is represented by all feasible parameters and their combinations. At the same time, all possible objective values represent an objective space.
        So, each point from the parameter space map to some point in objective space. The task for the estimator(s) is to provide a hypothesis that describes how this mapping performs.

        The task for the optimization algorithm is to find an optimal or perspective optimal region for all objectives. The optimization algorithm does this search based on a hypothesis from the estimator. In this context estimator(s) works as surrogate model for optimization.

        “In a dark place we find ourselves, and a little more knowledge lights our way” – Yoda
    
        1. define categorical columns -> transform pipeline as parameter
        2. fit -> generate surrogate(s) models for all parameters
        3. Find best categories with sampling strategies.
           Exampl: Generate 100 full-dim points -> Evaluate -> Take only categories from best samples. Fix it.
        4. Fix categories as mask for optimization algorithm. Categories become ortogonal for optimization problem
        5. For final predictions combine categories and population

    """

    def __init__(self,
                 bounds,
                 estimators,
                 train_test_sp=0.25,
                 solver='nsga2',
                 solver_gen=100,
                 solver_pop=100, # solver_params = dict(gen=100, pop=100)
                 cv_threshold='',
                 test_threshold='',
                 verbose=False
                ):
        """        
        Args:
            bounds (Tuple): Tuple with lower and higher bound for each feature in parameter space.
            portfolio : list of (str, estimator) tuples. Surrogates models to be verified. Defaults to None.
        """
        super().__init__(estimators=estimators)
        self._bounds = bounds
        self.verbose = verbose

        self._init_dataset: Tuple[list, list] = None
        self._train: Tuple[list, list] = None
        self._test: Tuple[list, list] = None
        self._candidates_uni = None
        self._score = ['r2', 'explained_variance',
                       'neg_mean_squared_error',
                       'neg_mean_absolute_error']

        self.cv_result = None  # resullts for each train/test folds
        # surrogates that pass cv validation and aggregation
        self.surr_valid = pd.DataFrame()
        self._solutions = None  # predict solutions from valid surrogates
        self._is_single: bool = False

        self._solver: str = solver
        self._pop_size: int = solver_pop
        self._gen: int = solver_gen

        self._train_test_sp: float = train_test_sp
        self._cv_threshold: str = cv_threshold
        self._test_threshold: str = test_threshold


        self._DATA_SET_MIN = None

        # dimensions mask
        # self._mask_col = mask_col,
        # self._mask_value = mask_val

    @property
    def bounds(self):
        return self._bounds

    @property
    def dataset(self):
        return self._init_dataset

    @property
    def models(self):
        return self._prf_models

    @property
    def solution(self):
        return self._solutions

    def get_name(self):
        names = [type(model).__name__.lower() for model in self._prf_models]
        return "Prediction Tutor for {}".format(names)

    def fit(self, X, y, validation: bool = True, **cv_params): 
        """ Fit the estimators.

        Args:
            X (array-like, sparse matrix) of shape (n_samples, n_features)
                Training vectors, where n_samples is the number of samples and
                n_features is the number of features.

            y (array-like, sparse matrix): array-like of shape (n_samples, n_outputs)
                Multi-output targets.

        Returns:
            self : object
        """        

        # --- Minimum count of samples for propper validation
        if self._DATA_SET_MIN is None:
            cv = cv_params['cv'] if 'cv' in cv_params else 5 # 5 folds default for sklearn
            MIN_TEST = 2  # min 2 test values for proper evalutio
            x = MIN_TEST*cv/(1-self._train_test_sp)
            self._DATA_SET_MIN = x + \
                ((MIN_TEST - x % MIN_TEST) if x % MIN_TEST else 0)

        X, y = check_X_y(X, y, 
                         multi_output=True,
                         accept_sparse=True,
                         estimator=self,
                         )
        self._init_dataset = (X, y) # ! ---------

        if validation:
            self._cv_fit(X, y, **cv_params)
        else:
            self._stack_fit(X, y, self.estimators)

        return self

    def _stack_fit(self, X, y, estimators, final_estimator = None):
        """ fit and stack selected estimators into one estimator
        """   

        if final_estimator:
            reg = StackingRegressor(
                estimators=estimators,
                final_estimator=final_estimator)
        else:
            reg = VotingRegressor(estimators)

        return reg.fit(X, y)


    def _cv_fit(self, X, y, **cv_params):

        if len(X) >= self._DATA_SET_MIN:
            # rewrite the previous dataset
            self._train = None
            self._test = None
            self._solutions = None
            self.surr_valid = pd.DataFrame()
            self.cv_result = None

            check_X_y(X, y, multi_output=True)
            self._is_single = y.shape[1] == 1

            logging.info(
                "Split dataset. Validation set is {}%".format(self._train_test_sp*100))
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self._train_test_sp)
            self._train = (X_train, y_train)
            self._test = (X_test, y_test)
        else:
            logging.info(
                "Available {}. At least {} samples are required".format(len(X), self._DATA_SET_MIN))
            return None

        # --- STAGE 1: Cross-validation surrogates
        cv_dataset = self._train
        cv_score = self.cross_validate(*cv_dataset, **cv_params)
        self.cv_result = self.cv_result_to_df(cv_score)

        # --- STAGE 2: Trashholding and aggregation.
        #              Refit valid models on train dataset
        self.surr_valid = self._results_aggregation(self.cv_result)
        if not self.surr_valid.empty:
            logging.info(
                "Stage-2: {} surrogate(s) valid".format(len(self.surr_valid)))
            self.surr_valid['estimator'].apply(
                lambda estimator: estimator.fit(*self._train))
            # --- STAGE 3: Score the aggregated surrogates with a test set.
            #              Sorting tham based on results
            self.surr_score(*self._test)
            if self._test_threshold:
                self.surr_valid.query(self._test_threshold, inplace=True)

            logging.info(
                "Stage-3: {} surrogate(s) pass surr score".format(len(self.surr_valid)))

            # --- STAGE 4: Generate solution from each valid hypothesis.
            #              Each valid hypothesis produce own solution(s).
            #              A Solution is validated based on a score metrics from stage 3.
            #              By default, a final prediction is a first-class prediction based on this ranging.
            #              If prediction have multiple solutions - By default, we take `n` solutions randomly from them.

            self._solutions = self.surr_valid.copy()
            # make a solver(s) on fitted surrogates (hypothesis)
            self._solutions['solver'] = self._solutions['estimator'].apply(
                self._construct_solver)
            # prediction vector(s) from solver(s)
            self._solutions['prediction'] = self._solutions['solver'].apply(
                lambda solver: solver.predict())
            # prediction vector(s) score. Hypervolume or min objective. Depends on solver. Less is better
            self._solutions['prediction_score'] = self._solutions['solver'].apply(
                lambda solver: solver.score())
        else:
            logging.info("Fit: Prediction from sampling plan")
            self._solutions = pd.DataFrame([{
                "model name": "sampling plan",
                "prediction": self._sobol(n=1)
            }])

    

    def cross_validate(self, X, y, **cv_params) -> dict:
        """ Select models that can describe a valid hypothesis for X, y """

        # -- 1. check parameters for cross-validation
        check_X_y(X, y, multi_output=True)
        if 'scoring' not in cv_params:
            cv_params['scoring'] = self._score
        cv_params['return_estimator'] = True
        # The result of cross-validation cannot be guaranteed for a smaller data set
        if len(X) < 8:
            return None

        # -- 2. check surrogates
        self._check_surr_candidate()

        # -- 3. cross-validation a surrogates as composition
        cv_score = self._candidates_uni.cross_validate(
            X, y, **cv_params)

        return cv_score

    def _check_surr_candidate(self) -> None:
        if not self._prf_models:
            raise ValueError(
                f"{self._prf_models}: Models portfolio is empty.")
        if self._candidates_uni is None:
            #!  What if dynamically change surrogates?
            self._candidates_uni = ModelsUnion(models=self._prf_models)

    def _thresholding(self, cv_result: pd.DataFrame) -> pd.DataFrame:
        # Reise Error in case of custom score metrics
        df = cv_result.copy()
        if len(self._init_dataset[0]) < 8:
            q = '(test_r2 > 0.97) & (test_neg_mean_absolute_error > -2.3) & (test_explained_variance > 0.6)'
            df.query(q, inplace=True)
        elif self._cv_threshold:
            # q = '(test_r2 > 0.65) & (test_neg_mean_absolute_error > -2.3) & (test_explained_variance > 0.6)'
            df.query(self._cv_threshold, inplace=True)

        return df

    def _results_aggregation(self, cv_result: pd.DataFrame) -> pd.DataFrame:

        # -- 1. Threshold validation the surrogates based on cv results
        models_df_score = self._thresholding(cv_result)
        if models_df_score.empty:
            return models_df_score

        # Constructs a new surrogates with the same parameters but that has not been fit on any data
        models_df_score.loc[:, 'estimator'] = models_df_score['estimator'].apply(
            clone)

        # Type 'Multi-objective': Surrogates that coverage all objective space
        all_y_models = models_df_score.loc[models_df_score.y_index == 'all'].drop_duplicates(
            subset='id')

        # -- 2. Aggregation
        # Type 'Single-objective': Surrogates that describe only 1 dimension from objective space
        splited_y_models = models_df_score.loc[models_df_score.y_index != 'all']

        # Make groups for each objective dimension. Sorting based on score
        grp = []
        y_names = []
        for name, group_df in splited_y_models.groupby(['y_index']):
            group_df_selection = group_df.drop_duplicates(subset='id')
            group_df_selection.sort_values(
                by=["test_" + s for s in self._score], ascending=False, inplace=True)
            grp.append(group_df_selection.reset_index(drop=True))
            y_names.append("meta_{}".format(name))

        # Absent valid surrogates in some dimension: we cannot restore the full object space
        # - return 'Multi-objective' type
        if self._init_dataset[1].shape[1] != len(grp):
            return all_y_models

        # There at least 1 valid surrogate for each objective dimension.
        # - We can combine groups to at least 1 valid hypothesis on objective space
        # - If there are multiple valid surrogates for some objective hypothesis we can duplicate
        #   other surrogates in other objective dimensions to full fill multiple hypotheses
        # - Compositional 'Multi-objective' surrogate(s) based on 'Single-objective'
        concat_y_groups = pd.concat(grp, axis=1, keys=y_names).fillna(
            method='backfill').fillna(method='ffill')

        # Make list from each cell & sum all dataframes in compositional 'Multi-objective' surrogates
        grp = [concat_y_groups[n].applymap(lambda x: [x]) for n in y_names]
        grp = sum(grp[1:], grp[0])
        grp.loc[:, 'estimator'] = grp['estimator'].apply(lambda models: ModelsUnion(models=clone(models),
                                                                                    split_y=True))
        grp.loc[:, 'id'] = grp['estimator'].apply(id)

        # Combine 'Multi-objective' and compositional 'Multi-objective' surrogates
        return pd.concat([all_y_models, grp], ignore_index=True)

    def surr_score(self, X=None, y=None) -> pd.DataFrame:
        """ Score a distill models on X, y test set and score on
        not dominated points from this test set """
        if X is None or y is None:
            raise ValueError("X and y are None")

        if self.surr_valid is not None:
            # Add columns with score metrics to dataframe with valid surrogates
            self.surr_valid.loc[:, 'surr_score'] = self.surr_valid['estimator'].apply(
                lambda estimator: estimator.score(*self._test))

            # none-dominated examples from test set
            ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(np.array(y))
            y_ndf_test = np.array(y)[ndf[0], :]
            X_ndf_test = np.array(X)[ndf[0], :]

            self.surr_valid.loc[:, 'ndf_surr_score'] = self.surr_valid['estimator'].apply(
                lambda estimator: estimator.score(X_ndf_test, y_ndf_test))

            # Sorting the solution(s). Score on the non dominated solution is more important
            self.surr_valid.sort_values(
                by=['ndf_surr_score', 'surr_score'], inplace=True, ascending=False)

            return self.surr_valid.copy()
        else:
            raise ValueError(
                "There is no valid surrogate model. {}".format(self.surr_valid))

    @staticmethod
    def cv_result_to_df(cv_score):
        """ Make DataFrame from cross validation results"""
        res = []

        def validate(data):
            if isinstance(data, dict):
                if 'fit_time' in list(data.values())[0]:
                    res.append(pd.DataFrame(list(data.values())[0]))
                else:
                    for _, v in data.items():
                        validate(v)
            if isinstance(data, list):
                for d in data:
                    validate(d)
        validate(cv_score)
        return pd.concat(res)

    def _construct_solver(self, estimator):
        """ Combine estimator(s) with solver"""
        if isinstance(estimator, ModelsUnion):
            models = estimator.models
        else:
            models = [estimator]

        solver = None
        if self._is_single is True:
            solver = Gaco(bounds=self._bounds).fit(models)
        elif self._solver == 'nsga2':
            solver = Nsga2(bounds=self._bounds,
                           pop_size=self._pop_size, gen=self._gen).fit(models)
        elif self._solver == 'moea_control':
            solver = MOEActr(
                bounds=self._bounds, pop_size=self._pop_size, gen=self._gen).fit(models)
        elif self._solver == 'random':
            solver = RandS(bounds=self._bounds, n=1000).fit(models)
        else:
            raise ValueError(
                "There is no exist solver {} ".format(self._solver))

        return solver

    def predict(self, X=None, y=None, n=1, kind='ndf_score', **predict_params):
        """ Prediction from the best solution.
        Solution get from solver, apply to the hypothesis """

        # --- STAGE 5: Manage predictions from valid surrogates.
        #              - Return predictions from first(best) surrogate model

        if self.surr_valid.empty:
            # -- 1. If surrogates not valid use sampling plan
            logging.info("There is no valid surrogates")
            pass
        else:
            if self._solutions is None:
                # -- 2. Error if surrogates valid but there is no solution
                raise AttributeError(
                    " A solution does not exist: {} \n\
                        Сheck that the surrogate models are fit appropriately".format(self._solutions))
            else:
                # -- 3. There is solution from valid surogates
                if kind == 'ndf_score':
                    sol_pool = self._sol_ndf_surr()
                elif kind == 'stack':
                    sol_pool = self._sol_stack_valid_surr()
                else:  # default
                   sol_pool = self._sol_ndf_surr()

                # Return `n` predictions
                if n is None:
                    return self._solutions['prediction'].values[0]
                else:
                    S = self._solutions['prediction'].values[0]
                    n = n if S.shape[0] > n else S.shape[0]
                    return S[np.random.choice(S.shape[0], n, replace=False), :]

        logging.info("Prediction from sampling plan")
        self._solutions = pd.DataFrame([{
            "model name": "sampling plan",
            "prediction": self._sobol(n=n)
        }])
        return self._solutions["prediction"][0].tolist()

    def _sol_ndf_surr(self):
        try:
            # get a predictions from best surrogate on none-dominated values
            pred = self._solutions.sort_values(
                by=['ndf_surr_score', 'surr_score'], ascending=False)['prediction'].values[0]
        except KeyError as err:
            # get the first predictions
            pred = self._solutions['prediction'].values[0]
            logging.info(
                " Sorting failed: {} \n{}".format(err, self._solutions))
            pass
        return pred

    def _sol_stack_valid_surr(self):
        pred_pools = self._solutions['prediction'].values
        return np.concatenate(pred_pools, axis=0)

    def predict_proba(self, X=None, **predict_params):
        """ Return all metrics of the best solution """
        if self._solutions is None:
            logging.info(
                " A solution does not exist: {}".format(self._solutions))
            return None
        else:
            return self._solutions

    def _sobol(self, n=1):
        """ Pseudorandom sampling from the search space """
        available = 0 if self._init_dataset is None else len(
            self._init_dataset[0])
        sbl = sobol_sample(self._bounds, n=available+n)[available:]
        return np.array(sbl)

    def _latin(self, n=1):
        """ Pseudorandom sampling from the search space """
        available = 0 if self._init_dataset is None else len(
            self._init_dataset[0])
        sbl = lh_sample(self._bounds, n=available+n)[available:]
        return sbl

    def _random(self, n=1):
        """ Random sampling from the search space """
        v_uniform = np.vectorize(uniform)
        return [v_uniform(*self._bounds) for _ in range(n)]


def sobol_sample(bounds, n=1):
    """ Sobol sampling

    Args:
        bounds (Tuple):  Tuple with lower and higher bound for each feature in objective space.
        Example: (([0., 0.]), ([2., 4.]))
        n (int, optional): Sample count. Defaults to 1.

    Returns:
        List: Point from search space
    """
    n_dim = len(bounds[0])
    sb = sobol_seq.i4_sobol_generate(n_dim, n, 1)
    diff = [r-l for l, r in zip(*bounds)]
    left = [l for l, _ in zip(*bounds)]
    return sb*diff+left


def lh_sample(bounds, n=1):
    """ Latin Hypercube sampling

    Args:
        bounds (Tuple):  Tuple with lower and higher bound for each feature in objective space.
        Example: (([0., 0.]), ([2., 4.]))
        n (int, optional): Sample count. Defaults to 1.

    Returns:
        List: Point from search space
    """
    n_dim = len(bounds[0])
    h_cube = np.random.uniform(size=[n, n_dim])
    for i in range(0, n_dim):
        h_cube[:, i] = (np.argsort(h_cube[:, i])+0.5)/n
    diff = [r-l for l, r in zip(*bounds)]
    left = [l for l, _ in zip(*bounds)]
    return h_cube*diff+left



def _name_estimators(estimators):
    """Generate names for estimators."""

    names = [
        estimator
        if isinstance(estimator, str) else type(estimator).__name__.lower()
        for estimator in estimators
    ]
    namecount = defaultdict(int)
    for est, name in zip(estimators, names):
        namecount[name] += 1

    for k, v in list(namecount.items()):
        if v == 1:
            del namecount[k]

    for i in reversed(range(len(estimators))):
        name = names[i]
        if name in namecount:
            names[i] += "-%d" % namecount[name]
            namecount[name] -= 1

    return list(zip(names, estimators))
