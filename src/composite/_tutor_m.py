from itertools import cycle
from typing import Mapping, MutableMapping, List, Tuple
import hashlib
import logging
from copy import deepcopy
from random import uniform
from collections import OrderedDict, defaultdict

import pygmo as pg
import pandas as pd
import sobol_seq
import numpy as np

# -- utilities
from sklearn import clone
from sklearn.model_selection import cross_validate
from sklearn.base import TransformerMixin, BaseEstimator, MetaEstimatorMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted, check_X_y, _check_fit_params
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from joblib import Parallel, delayed
import category_encoders as ce

from src.surrogate.surrogate_union import Union

def get_bounds(desc, params_enc):
    bounds = []
    kv_desc = {p['name']: p for p in desc}
    for col in params_enc.columns:
        if 'categories' in kv_desc[col]:
            bounds.append(list(range(1, len(kv_desc[col]['categories'])+1)))
        else:
            bounds.append(kv_desc[col]['bounds'])

    return list(zip(*bounds))


class PygmoProblem():

    def __init__(self,
                 surrogate,
                 bounds, 
                 integer):
        self._surrogate = surrogate
        self._bounds = bounds
        self._int = integer

    def fitness(self, x):
        x = np.array(x)
        result = self._surrogate.predict(x.reshape(1, -1))
        return result.flatten().tolist()

    def get_nobj(self):
        prediction = self._surrogate.predict([self._bounds[0]])
        nobj = prediction.flatten().size
        return nobj

    # Return bounds of decision variables
    def get_bounds(self):
        return self._bounds

    # Integer Dimension
    def get_nix(self):
        return self._int

    def gradient(self, x):
        return pg.estimate_gradient_h(lambda x: self.fitness(x), x)

    # Return function name
    def get_name(self):
        meta_name = type(self._surrogate).__name__
        if hasattr(self._surrogate, 'surrogates'):
            return self._surrogate._union_type + "(" + "+".join([type(t).__name__ for t in self._surrogate.surrogates]) + ")"
        else:
            return meta_name



class BaseTutor():
    def __init__(self, 
                 surrogates: List,
                 optimizers: List = None,
                 union_type: str = None,
                 **parameters):
        self._surrogates = surrogates
        self._optimizers = optimizers
        self._union_type = union_type
        self.f_desc = None
        self.obj_desc = None
        self._prob = None

        self._encoder = None
        self._params_enc = None
        self._obj_select = None

    def build_model(self, params, obj, params_desc, **kargs):
        """ Build surrogate model(s) from available samples

            params, obj - pandas DataFrame
        """
        self.OBJECTIVES = ['test_roc_auc', 'fit_time']
        self.OBJECTIVES_DESC = {
            "fit_time": -1,
            "score_time": -1,
            "test_f1": 1,
            "test_roc_auc": 1
        }

        self._params_enc = None
        self._obj_select = None
        self._prob = None

        # [1] --- Check samples
        # X, y = check_X_y(X, y,
        #                 multi_output=True,
        #                 ensure_min_samples=5)
        # TODO:validate entropy in parameters

        # [2] --- Search-space and objectives description
        # self.f_desc = features_desc
        # self.obj_desc = OBJECTIVES_DESC

        # [3] --- Encode categorical black-box parameters
        cat_columns = params.select_dtypes(include=[object]).columns
        self._encoder = ce.OrdinalEncoder(cols=cat_columns)

        params_enc = self._encoder.fit_transform(params)
        params_enc.sort_index(axis=1,
                              ascending=False,
                              key=lambda c: params_enc[c].dtypes,
                              inplace=True)

        self._params_enc = params_enc

        # [4] --- Select objectives and transform maximization objectives into minimization one
        obj_sign_vector = []
        for c in obj.columns:
            if c in self.OBJECTIVES_DESC.keys():
                obj_sign_vector.append(self.OBJECTIVES_DESC[c])
            else:
                obj_sign_vector.append(-1)
                logging.warning(
                    f"There is no configuration for objective {c}. Default is minimization"
                )
        # inverse. maximization * -1 == minimization
        self._obj_select = (obj * -1 * obj_sign_vector)[self.OBJECTIVES]

        # [5] --- Fit surrogate model with black-box parameters
        if self._union_type is None and len(self._surrogates) == 1:
            model = deepcopy(self._surrogates[0])
        else:
            model = Union(surrogates=self._surrogates,
                          union_type=self._union_type)

        model.fit(self._params_enc.values, self._obj_select.values)

        # [6] --- Create optimization problem that based on surrogate model
        bounds = get_bounds(params_desc, self._params_enc)
        integer_params = (self._params_enc.dtypes == np.int64).sum()
        self._prob = pg.problem(PygmoProblem(model, bounds, integer_params))

        return self

    def predict(self, n=None):
        if self._prob is None:
            raise ValueError(
                f"Before prediction {self.__class__.__name__} requires first to build surrogate model. \
                Please check if your surrogate model is built.")

        # [7] --- Optimization
        isl = pg.island(algo=pg.nsga2(gen=300),  # NSGA2
                        size=100,
                        prob=self._prob,
                        udi=pg.mp_island(),
                        )
        isl.evolve()
        isl.wait()
        final_pop = isl.get_population()

        idx = pg.fast_non_dominated_sorting(final_pop.get_f())[0][0]
        pop_df = pd.DataFrame(
            final_pop.get_x()[idx],
            columns=self._params_enc.columns)
        pop_df['n_estimators'] = pop_df['n_estimators'].astype('int32') #! only for RF parameters
        propos = self._encoder.inverse_transform(pop_df)

        if n is None:
            return propos
        else:
            n = n if propos.shape[0] > n else propos.shape[0]
        return propos.sample(n=n)

    def score(self, param_test, score_true, callback_func):
        if self._prob is None:
            raise ValueError(
                f"Before score {self.__class__.__name__} requires first to build surrogate model. \
                Please check if your surrogate model is built.")

        param_test = self._encoder.fit_transform(param_test) #! encode parameters
        score_pred = pd.DataFrame(
            [self._prob.fitness(p) for p in param_test.values],
            columns=self.OBJECTIVES)
        
        return [callback_func(*obj) for obj in zip(score_true[self.OBJECTIVES].T.values, score_pred.T.values)]

class TutorM():
    def __init__(self,
                 portfolio: List,
                 train_test_sp=0.25,
                 cv_threshold=0,
                 test_threshold=0,
                 ):

        self.portfolio = portfolio
        self._init_dataset: Tuple[list, list] = None
        self._train: Tuple[list, list] = None
        self._test: Tuple[list, list] = None

        self._cv_result = None  # resullts for each train/test folds
        self._test_result = None  # surrogates that pass cv validation and aggregation
        self._solutions = None  # predict solutions from valid surrogates

        self._train_test_sp: float = train_test_sp
        self._cv_threshold: float = cv_threshold
        self._test_threshold: float = test_threshold

        self._status = {
            'cv': None,
            'test': None,
            'solutions': None
            'predict': None 
        }

    def build_model(self, params, obj, params_desc, **kargs):

        self._status = {
            'cv': None,
            'test': None,
            'solutions': None
            'predict': None
        }

        self.OBJECTIVES = ['test_roc_auc', 'fit_time']
        X_cv, X_test, y_cv, y_test = train_test_split(
            params, obj[self.OBJECTIVES], test_size=self._train_test_sp)

        # --- STAGE 1: surrogate combinations
        surr_comb = [BaseTutor(comb, union_type='separate')
                        for comb in zip(*[self.portfolio]*2)]

        with Parallel(prefer='threads') as parallel:
            # --- STAGE 2: cross-validate (cv) combinations
            cv_score = []
            for train_idx, valid_idx in KFold(n_splits=3).split(X_cv):
                comb_copy = deepcopy(surr_comb)
                tutors = parallel(delayed(tutor.build_model)(X_cv.iloc[train_idx], y_cv.iloc[train_idx], params_desc)
                                for tutor in comb_copy)

                score = parallel(delayed(tutor.score)(X_cv.iloc[valid_idx], y_cv.iloc[valid_idx], r2_score)
                                for tutor in comb_copy)

                cv_score.append(score)
            # score from cv
            cv_score_mean = np.array(cv_score).mean(axis=0)
            surr_cv_names = np.array([[tutor._prob.get_name()] for tutor in comb_copy])
            cv_surr = pd.DataFrame(np.concatenate(
                [surr_cv_names, cv_score_mean], 
                axis=1, 
                dtype=np.object),
                columns=['surrogate']+self.OBJECTIVES)
            self._status['cv'] = cv_surr
            cv_pass = (cv_surr[self.OBJECTIVES] > self._cv_threshold).assign(
                                      surr=deepcopy(self.portfolio)
                                      )
            
            if (cv_pass[self.OBJECTIVES].sum() > 0).all() # check that at least for each objective there is one valid surrogate
                cv_valid_surr = []
                for obj in self.OBJECTIVES:
                    surr_for_obj = cv_pass.query(f"{obj}==True").values
                    cv_valid_surr.append(surr_for_obj)

                surr_ext = []
                max_count = max([len(surr) for surr in cv_valid_surr])
                for surr_for_obj in cv_valid_surr:
                    if len(surr_for_obj) == max_count:
                        surr_ext.append(surr_for_obj)
                    else:
                        surr_ext.append(cycle(surr_for_obj))         
                surr_comb_cv = [BaseTutor(comb, union_type='separate') for comb in list(
                    zip(surr_ext))]  # surrogates combinations that pass cv
            else:
                return self  # surrogates can't be used


            # --- STAGE 3: retrain and test valid combinations from cross-validation
            surr_test = deepcopy(surr_comb_cv)
            tutors = parallel(delayed(tutor.build_model)(X_cv, y_cv, params_desc)
                              for tutor in surr_test)
            test_score = parallel(delayed(tutor.score)(X_test, y_test, r2_score)
                             for tutor in surr_test)

            test_surr_names = [tutor._prob.get_name() for tutor in surr_test]

            test_surr = pd.DataFrame(np.concatenate(
                [test_surr_names, test_score],
                axis=1,
                dtype=np.object),
                columns=['surrogate']+self.OBJECTIVES)

            self._status['test'] = test_surr
            test_pass = (test_surr[self.OBJECTIVES] > self._test_threshold).assign(
                surr=deepcopy(surr_comb_cv)
            )

                        # check that at least for each objective there is one valid surrogate
            if (test_pass[self.OBJECTIVES].sum() > 0).all()
                # surr_comb_test = # TODO: select test samples
            else:
                return self  # surrogates can't be used


        # --- STAGE 4: retrain valid combinations from test


    def predict(self, n=None):
        pass
        # --- STAGE 5: select predictions


    def _stack_fit(self, X, y, estimators, final_estimator=None):
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
