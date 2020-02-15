from typing import List, Tuple
import hashlib
from random import uniform

import pygmo as pg
import pandas as pd
import sobol_seq

import numpy as np
from sklearn import clone
from sklearn.model_selection import cross_validate
from sklearn.base import TransformerMixin, BaseEstimator, MetaEstimatorMixin
from sklearn.utils.validation import check_is_fitted, check_X_y
from sklearn.model_selection import train_test_split


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel, RationalQuadratic, ExpSineSquared
import sklearn.gaussian_process as gp

from joblib import Parallel, delayed

from .search import Gaco, Nsga2

np.random.seed(1)

TRAIN_TEST_SPLIT = 0.25


class HypothesisTransformer(TransformerMixin, MetaEstimatorMixin, BaseEstimator):
    """ Allows using an estimator such as a transformator in an earlier step of a pipeline
    """

    def __init__(self, estimator: BaseEstimator, predict_func: str = 'predict'):
        """
        Args:
            estimator (BaseEstimator): An instance of the estimator that should be used for the transformation.
            predict_func (str, optional): Function of the estimator that will be used as transformer. Defaults to 'predict'.
        """
        self.estimator = estimator
        self.estimator_ = None
        self.predict_func = predict_func

    def fit(self, X, y):
        X, y = check_X_y(X, y, estimator=self)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        X, y = check_X_y(X, y, estimator=self)

        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y, **fit_params)
        return [self.estimator_]

    def transform(self, X=None):
        """ Applies the `predict_func` on the fitted estimator.
        Returns:
            [Array]: shape `(X.shape[0], )`
        """
        check_is_fitted(self, 'estimator_')
        return getattr(self.estimator_, self.predict_func)(X).reshape(-1, 1)


class ModelsUnion(TransformerMixin, BaseEstimator):
    """ Allows parallel fitting several models on the same
    feature space(X) with splitting in objective space(y).

    Args:
        TransformerMixin: Mixin class for all transformers in scikit-learn.
        BaseEstimator: Base class for all estimators in scikit-learn.

    Raises:
        Exception: In case of splitin objective space, strict count of models
        are not equal to the dimensionality of objective space.
        Strict mean more than one. One model scales to the required dimension.

    Returns:
        [type]: [description]
    """

    def __init__(self, models: List[BaseEstimator], n_jobs: int = -1, split_y: bool = False):
        self._models = models
        self.n_jobs = n_jobs
        self.split_y = split_y

    @property
    def models(self):
        return self._models

    @models.setter
    def models(self, models: List[BaseEstimator]):
        self._models = models

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, *fit_params)
        return self._models

    def transform(self, X, *arg, **kwargs):
        return X

    def fit(self, X, y=None, **fit_params):
        self._check_y_models(y)
        self._models = self._parallel_func(X, y, fit_params, _fit)
        return self

    def predict(self, X, **predict_params):
        """Predicts target using underlying estimators"""
        pred_result = self._parallel_func(X, None, predict_params, _predict)

        if len(self._models) == 1:
            pred_result = pred_result[0]
        elif self.split_y:
            pred_result = np.vstack(pred_result).T.tolist()
        else:
            pred_result = np.array(pred_result).mean(axis=0).tolist()

        return np.array(pred_result)

    def cross_validate(self, X, y, **cv_params):
        self._check_y_models(y)
        return self._parallel_func(X, y, cv_params, _cv)

    def score(self, X, y=None, **score_params):
        score_result = self._parallel_func(X, y, score_params, _score)
        if len(self._models) == 1:
            score_result = score_result[0]
        else:
            score_result = np.array(score_result).mean()

        return score_result

    def _check_y_models(self, y):
        """ Checking the conformity of models with objective-space

        Args:
            y (1D or 2D array): Objective-space
        """
        if self._models is None:
            Exception("There are no models")

        if self.split_y:
            if len(self._models) == 1:
                print("Expand the model count for each label. Clone: {}".format(
                    type(self._models[0]).__name__))
                etalon = self._models[0]
                self._models = [clone(etalon) for _ in range(y.shape[1])]
            elif y.shape[1] == len(self._models):
                pass
            else:
                raise Exception(
                    "Count of models is not equal to the dimensionality of objective space.\n Models: {}\
                    You can reduce the number of models to one or equalize the number \
                    of models to the dimensions of objective space. \
                    [{}] dimension in y-space vs [{}] models length".format(self._models, len(y), len(self._models)))

    def _parallel_func(self, X, y, func_params, func):
        """Runs func in parallel on estimators"""
        return Parallel(n_jobs=self.n_jobs)(delayed(func)(
            model, X, y, split_y=self.split_y,
            message_clsname='_ModelsUnion',
            idx=idx,
            message='Parallel contex. idx:{}, length:{}'.format(
                idx, len(self._models)),
            **func_params) for idx, model in enumerate(self._models, 0))


def _cv(estimator, X, y, split_y=False, message_clsname='', idx=0, message=None, **cv_params):
    """ Cross-validation the single estimator.

    Args:
        estimator (BaseEstimator): Singl estimator for cross validation
        X (2D array): Feature-space
        y (1D or 2D array): Objective-space
        split_y (bool, optional): Spliting objective space. Defaults to False.
        message_clsname (str, optional): Log message. Defaults to ''.
        idx (int, optional): Model positional index. In the case of y-space allocation,
            it corresponds to the evaluation on appropriate objective index. Defaults to 0.
        message (str, optional): Log message. Defaults to None.

    Returns:
        Dict: Cross-validation results
    """
    X, y = check_X_y(X, y, estimator=estimator, multi_output=True)
    if split_y:
        y = y[:, idx]
    X, y = check_X_y(X, y, multi_output=True, estimator=estimator)
    y_index = idx if split_y else "all"

    if hasattr(estimator, 'cross_validate'):
        cv_result = estimator.cross_validate(X, y, **cv_params)
    else:
        cv_result = cross_validate(estimator, X, y, **cv_params)
        cv_result['y_index'] = y_index
        cv_result['model name'] = type(estimator).__name__.lower()
        cv_result['params hash'] = hashlib.md5(
            str(estimator.get_params()).encode('utf-8')).hexdigest()[:4]

        ident = cv_result['model name']+cv_result['params hash']
        cv_result['id'] = hashlib.md5(
            ident.encode('utf-8')).hexdigest()[:6]

    key = "{}_{}".format(y_index, type(estimator).__name__.lower())
    return {key: cv_result}


def _fit(transformer, X, y, split_y=False, message_clsname='', idx=0, message=None, **fit_params):
    """ Fitting the single estimator

    Args:
        estimator (BaseEstimator): Singl estimator for fitting.
        X (2D array): Feature-space
        y (1D or 2D array): Objective-space
        split_y (bool, optional): Spliting objective space. Defaults to False.
        message_clsname (str, optional): Log message. Defaults to ''.
        idx (int, optional): Model positional index. In the case of y-space allocation,
            it corresponds to the evaluation on appropriate objective index. Defaults to 0.
        message (str, optional): Log message. Defaults to None.

    Returns:
        List or Number: Fit result
    """
    X, y = check_X_y(X, y, estimator=transformer, multi_output=True)
    if split_y:
        y = y[:, idx]
    X, y = check_X_y(X, y, multi_output=True, estimator=transformer)
    return transformer.fit(X, y, **fit_params)


def _score(estimator, X, y, split_y=False, message_clsname='', idx=0, message=None, **score_params):
    """ Score evaluation on the single estimator

    Args:
        estimator (BaseEstimator): Singl estimator for score evaluation.
        X (2D array): Feature-space.
        y (1D or 2D array): Objective-space.
        split_y (bool, optional): Spliting objective space. Defaults to False.
        message_clsname (str, optional): Log message. Defaults to ''.
        idx (int, optional): Model positional index. In the case of y-space allocation,
            it corresponds to the evaluation on appropriate objective index. Defaults to 0.
        message (str, optional): Log message. Defaults to None.

    Returns:
        List or Number: Score result
    """
    X, y = check_X_y(X, y, estimator=estimator, multi_output=True)
    if split_y:
        y = y[:, idx]
    X, y = check_X_y(X, y, multi_output=True, estimator=estimator)
    return estimator.score(X, y, **score_params)


def _predict(estimator, X, y=None, split_y=False, message_clsname='', idx=None, message=None, **predict_params):
    """ Prediction from the single estimator

    Args:
        estimator (BaseEstimator): Singl estimator for prediction.
        X (2D array): Feature-space.
        y (1D or 2D array): Objective-space. Defaults to False.
        split_y (bool, optional): Spliting objective space. Defaults to False.
        message_clsname (str, optional): Log message. Defaults to ''.
        idx (int, optional): Model positional index. In the case of y-space allocation,
            it corresponds to the evaluation on appropriate objective index. Defaults to 0.
        message (str, optional): Log message. Defaults to None.

    Returns:
        List or Number: Prediction
    """
    print(message_clsname, message)
    return estimator.predict(X, **predict_params).tolist()


class PredictTutor(BaseEstimator):
    """ Agregate and sort best preidctions from multiple models."""

    def __init__(self, bounds, portfolio: List[BaseEstimator] = None):
        """        
        Args:
            bounds (Tuple): Tuple with lower and higher bound for each feature in objective space.
            portfolio (List[BaseEstimator], optional): a list of surrogates to be verified. Defaults to None.
        """
        self.__bounds = bounds
        self.__init_dataset: Tuple[list, list] = None
        self.__prf_models = portfolio

        self._train: Tuple[list, list] = None
        self._test: Tuple[list, list] = None
        self._candidates_uni = None
        self._score = ['r2', 'explained_variance',
                       'neg_mean_squared_error',
                       'neg_mean_absolute_error']

        self.cv_result = None  # resullts for each train/test folds
        self.surr_valid = pd.DataFrame()  # surrogates that pass cv validation and aggregation
        self._solutions = None  # predict solutions from valid surrogates
        self._is_single: bool = False

    @property
    def bounds(self):
        return self.__bounds

    @property
    def dataset(self):
        return self.__init_dataset

    @property
    def models(self):
        return self.__prf_models

    @property
    def solution(self):
        return self._solutions

    def get_name(self):
        names = [type(model).__name__.lower() for model in self.__prf_models]
        return "Prediction Tutor for {}".format(names)

    def fit(self, X, y, **cv_params):
        """ Fit new dataset to class instance.
            Split the dataset to train and test if there are enough experiments.
        """
        if X is None or y is None:
            return None

        self.__init_dataset = (X, y)

        # There is no validation for set smaller than 8.
        # 25% of the data set is less than 2 test values for proper scoring.
        if len(X) >= 8:
            # rewrite the previous dataset
            self._train = None
            self._test = None
            self._solutions = None
            self.surr_valid = pd.DataFrame()
            self.cv_result = None

            check_X_y(X, y, multi_output=True)
            self._is_single = y.shape[1] == 1

            print("Split dataset. Validation set is {}%".format(TRAIN_TEST_SPLIT))
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TRAIN_TEST_SPLIT)
            self._train = (X_train, y_train)
            self._test = (X_test, y_test)
        else:
            print("Available {}. At least 8 samples are required".format(len(X)))
            return None

        # --- STAGE 1: Cross-validation surrogates
        cv_dataset = self._train or self.__init_dataset
        cv_score = self.cross_validate(*cv_dataset, **cv_params)
        self.cv_result = self.cv_result_to_df(cv_score)

        # --- STAGE 2: Trashholding and aggregation.
        #              Refit valid models on train dataset
        self.surr_valid = self._results_aggregation(self.cv_result)
        if not self.surr_valid.empty:
            print("{} surrogate(s) valid".format(len(self.surr_valid)))
            self.surr_valid['estimator'].apply(
                lambda estimator: estimator.fit(*self._train))

        # --- STAGE 3: Score the aggregated surrogates with a test set.
        #              Sorting tham based on results
        self._solutions = self.surr_score(*self._test)

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

    def cross_validate(self, X, y, **cv_params) -> dict:
        """ Select models that can describe a valid hypothesis for X, y """

        # -- 1. check parameters for cross-validation
        check_X_y(X, y, multi_output=True)
        if 'scoring' not in cv_params:
            cv_params['scoring'] = self._score
        cv_params['return_estimator'] = True
        # The result of cross-validation cannot be guaranteed for a smaller data set
        if len(X) < 4:
            return None

        # -- 2. check surrogates
        self._check_surr_candidate()

        # -- 3. cross-validation a surrogates as composition
        cv_score = self._candidates_uni.cross_validate(
            X, y, **cv_params)

        return cv_score

    def _check_surr_candidate(self) -> None:
        if not self.__prf_models:
            raise ValueError(
                f"{self.__prf_models}: Models portfolio is empty.")
        if self._candidates_uni is None:
            #!  What if dynamically change surrogates?
            self._candidates_uni = ModelsUnion(models=self.__prf_models)

    def _thresholding(self, cv_result: pd.DataFrame) -> pd.DataFrame:
        # Reise Error in case of custom score metrics
        if len(self.__init_dataset[0]) < 12:
            q = '(test_r2 > 0.97) & (test_neg_mean_absolute_error > -2.3) & (test_explained_variance > 0.6)'
        else:
            q = '(test_r2 > 0.75) & (test_neg_mean_absolute_error > -2.3) & (test_explained_variance > 0.6)'
        return cv_result.query(q)

    def _results_aggregation(self, cv_result: pd.DataFrame) -> pd.DataFrame:

        # -- 1. Threshold validation the surrogates based on cv results
        models_df_score = self._thresholding(cv_result)
        if models_df_score.empty:
            return models_df_score

        # Constructs a new surrogates with the same parameters but that has not been fit on any data
        models_df_score['estimator'] = models_df_score['estimator'].apply(
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
        if len(self.__init_dataset[1].columns) != len(grp):
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
        grp['estimator'] = grp['estimator'].apply(lambda models: ModelsUnion(models=clone(models),
                                                                             split_y=True))
        grp['id'] = grp['estimator'].apply(id)

        # Combine 'Multi-objective' and compositional 'Multi-objective' surrogates
        return pd.concat([all_y_models, grp], ignore_index=True)

    def surr_score(self, X=None, y=None) -> pd.DataFrame:
        """ Score a distill models on X, y test set and score on
        not dominated points from this test set """
        if X is None or y is None:
            raise ValueError("X and y are None")

        if self.surr_valid is not None:
            # Add columns with score metrics to dataframe with valid surrogates
            self.surr_valid['val_score'] = self.surr_valid['estimator'].apply(
                lambda estimator: estimator.score(*self._test))

            # none-dominated examples from test set
            ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(y.values)
            y_ndf_test = y.iloc[ndf[0], :]
            X_ndf_test = X.iloc[ndf[0], :]
            self.surr_valid['ndf_val_score'] = self.surr_valid['estimator'].apply(
                lambda estimator: estimator.score(X_ndf_test, y_ndf_test))

            # Sorting the solution(s). Score on the non dominated solution is more important
            self.surr_valid.sort_values(
                by=['ndf_val_score', 'val_score'], inplace=True, ascending=False)

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
            solver = Gaco(bounds=self.__bounds).fit(models)
        else:
            solver = Nsga2(bounds=self.__bounds).fit(models)
        return solver

    def predict(self, X=None, y=None, n=1, **predict_params):
        """ Prediction from the best solution.
        Solution get from solver, apply to the hypothesis """

        # --- STAGE 5: Manage predictions from valid surrogates.
        #              - Return predictions from first(best) surrogate model

        if self.surr_valid.empty:
            # -- 1. If surrogates not valid use sampling plan
            pass
        else:
            # -- 2. Error if surrogates valid but there is no solution
            if self._solutions is None:
                raise AttributeError(
                    " A solution does not exist: {} \n\
                    Сheck that the surrogate models are fit appropriately".format(self._solutions))
            else:
                # -- 3. There is solution from valid surogates. Return `n` predictions 
                if n is -1:
                    return self._solutions['prediction'].values[0]
                else:
                    S = self._solutions['prediction'].values[0]
                    n = n if S.shape[0] > n else S.shape[0]
                    return S[np.random.choice(S.shape[0], n, replace=False), :]
        print("Prediction from sampling plan")     
        return self._sobol(n=n)

    def next_config(self, X, y, n=1, **cv_params):
        if X is not None and y is not None:
            is_valid = self.cross_validate(X, y, **cv_params)
            if is_valid:
                self._generate_solution()
                return self.predict(n=n)
            else:
                return self._sobol(n=n)
        else:
            return self._sobol(n=n)

    def predict_proba(self, X=None, **predict_params):
        """ Return all metrics of the best solution """
        if self._solutions is None:
            print(" A solution does not exist: {} \n\
                Сheck that the surrogate models are fit appropriately".format(self._solutions))
            return None
            # raise AttributeError(
            #     " A solution does not exist: {} \n\
            #       Сheck that the surrogate models are fit appropriately".format(self._solutions))
        else:
            return self._solutions

    def _sobol(self, n=1):
        """ Pseudorandom sampling from the search space """
        available = 0 if self.__init_dataset is None else len(
            self.__init_dataset[0])
        sbl = sobol_sample(self.__bounds, n=available+n)[available:]
        return sbl

    def _latin(self, n=1):
        """ Pseudorandom sampling from the search space """
        available = 0 if self.__init_dataset is None else len(
            self.__init_dataset[0])
        sbl = sobol_sample(self.__bounds, n=available+n)[available:]
        return sbl

    def _random(self, n=1):
        """ Random sampling from the search space """
        v_uniform = np.vectorize(uniform)
        return [v_uniform(*self.__bounds) for _ in range(n)]


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
