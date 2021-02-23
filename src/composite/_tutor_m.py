from itertools import cycle
from typing import List, Tuple
import logging
from copy import deepcopy
from random import uniform

import pygmo as pg
import pandas as pd
import sobol_seq
import numpy as np
import category_encoders as ce

from sklearn.utils.validation import check_is_fitted, check_X_y, _check_fit_params
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from joblib import Parallel, delayed

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
        self.obj_sign_vector = []
        for c in obj.columns:
            if c in self.OBJECTIVES_DESC.keys():
                self.obj_sign_vector.append(self.OBJECTIVES_DESC[c])
            else:
                self.obj_sign_vector.append(-1)
                logging.warning(
                    f"There is no configuration for objective {c}. Default is minimization"
                )
        # inverse. maximization * -1 == minimization
        self._obj_select = (obj * -1 * self.obj_sign_vector)[self.OBJECTIVES]

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
        score = np.array([self._prob.fitness(
            p) for p in param_test.values]) * -1 * self.obj_sign_vector #! change sign of an objective

        # score_pred = pd.DataFrame(
        #     score,
        #     columns=self.OBJECTIVES)
        
        return [callback_func(*obj) for obj in zip(score_true[self.OBJECTIVES].T.values, score.transpose())]

class TutorM():
    def __init__(self,
                 portfolio: List,
                 train_test_sp=0.25,
                 cv_threshold=-15,
                 test_threshold=-15,
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
            'final_surr': None,
            'solutions': None,
            'predict': None 
        }

    def build_model(self, params, obj, params_desc, **kargs):

        self._status = {
            'cv': None,
            'test': None,
            'final_surr': None,
            'solutions': None,
            'predict': None
        }
        self._params_desc = params_desc
        self._samples_count = len(params)
        self.OBJECTIVES = ['test_roc_auc', 'fit_time']
        X_cv, X_test, y_cv, y_test = train_test_split(
            params, obj[self.OBJECTIVES], test_size=self._train_test_sp)

        # --- STAGE 1: surrogate combinations
        surr_comb = [BaseTutor(comb, union_type='separate')
                        for comb in zip(*[self.portfolio]*2)]
        logging.info(f"[1] There are {len(self.portfolio)} surrogates in portfolio. ")

        with Parallel(prefer='threads') as parallel:
            # --- STAGE 2: cross-validate(cv) combinations
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
            
            if (cv_pass[self.OBJECTIVES].sum() > 0).all(): # check that at least for each objective there is one valid surrogate
                cv_valid_surr = []
                for obj_name in self.OBJECTIVES:
                    surr_for_obj = cv_pass.query(f"{obj_name}==True").sort_values(
                        by=[obj_name], ascending=False)['surr'].values
                    cv_valid_surr.append(surr_for_obj)

                surr_ext = []
                max_count = max([len(surr) for surr in cv_valid_surr])
                for surr_for_obj in cv_valid_surr:
                    if len(surr_for_obj) == max_count:
                        surr_ext.append(surr_for_obj)
                    else:
                        surr_ext.append(cycle(surr_for_obj))         
                surr_comb_cv = [BaseTutor(comb, union_type='separate')
                                for comb in zip(*surr_ext)]  # surrogates combinations that pass cv

                logging.info(f"[2] There are {max_count} valid surrogate combinations. ")
            else:
                logging.info(f"[2] There is no complete set of valid surrogates for each objective after cross-validation. ")
                return self  # surrogates can't be used


            # --- STAGE 3: retrain and test valid combinations from cross-validation
            tutor_test = deepcopy(surr_comb_cv)
            tutors = parallel(delayed(tutor.build_model)(X_cv, y_cv, params_desc)
                              for tutor in tutor_test)
            test_score = parallel(delayed(tutor.score)(X_test, y_test, r2_score)
                             for tutor in tutor_test)
            test_score_mean = np.mean(test_score, axis=1, keepdims=True)

            test_surr_names = [[tutor._prob.get_name()] for tutor in tutor_test]

            test_surr = pd.DataFrame(np.concatenate(
                [test_surr_names, test_score_mean],
                axis=1,
                dtype=np.object),
                columns=['surrogate', 'r2'])
            
            test_surr.sort_values(by=['r2'], ascending=False, inplace=True)

            test_surr['pass'] = test_surr['r2'] > self._test_threshold
            test_surr['tutors'] = tutors
            self._status['test'] = test_surr

            # check that at least there is one valid combination
            if (test_surr['pass'] == True).sum() > 0:
                test_valid_surr = test_surr[test_surr['pass'] == True]['tutors'].values
                logging.info(f"[3] There are {len(test_valid_surr)} valid surrogate combinations. ")
            else:
                logging.info(f"[3] There is no valid surrogate set after test stage. ")
                return self  # surrogates can't be used


            # --- STAGE 4: retrain valid combinations from test
            final_surr = parallel(delayed(tutor.build_model)(params, obj[self.OBJECTIVES], params_desc)
                                  for tutor in test_valid_surr)
            self._status['final_surr'] = final_surr
            logging.info(f"[4] There are {len(final_surr)} valid surrogate combinations. ")

            return self
            

    def predict(self, n=10, kind='stack'):
        # --- STAGE 5: select predictions
        if self._status['final_surr'] is None:
            logging.info(f"[5] There is no valida surrogates. Prediction from the pseudo-random sampler. ")
            
            # how many samples were used in the build phase
            s = self._samples_count if self._samples_count else 0
            prediction_pool = [self._psd_sample(n=s+n)[s:]]
        else:
            if kind=='stack':
                prediction_pool = [tutor.predict(n) for tutor in self._status['final_surr']]
            elif kind == 'best':
                prediction_pool = [self._status['final_surr'][0].predict(n)]

        solutions = pd.concat(prediction_pool)
        n = n if len(solutions) > n else len(solutions)
        logging.info(
            f"Predict {n} configurations fron {len(solutions)} in the pool. ")

        self._status['solutions'] = solutions

        return solutions.sample(n=n)


    def _psd_sample(self, n=1, kind='sobol'):
        """ Pseudo-random configuration sampling

        Args:
            n (int, optional): number of configurations to generate. Defaults to 1.
            kind (str, optional): identifying the general kind of pseudo-random samples generator. Sobol('sobol') or Latin Hypercube sampling('lh'). Defaults to 'sobol'.

        Raises:
            ValueError: wronf value for pseudo-samples generation

        Returns:
            pandas.DataFrame: configurations samples
        """

        dim = len(self._params_desc)
        if kind == 'sobol':
            squere = sobol_seq.i4_sobol_generate(dim, n)
        elif kind == 'lh':
            squere = np.random.uniform(size=[n, dim])
            for i in range(0, dim):
                squere[:, i] = (np.argsort(squere[:, i])+0.5)/n
        else:
            raise ValueError(
                "`kind` property could be `sobol` or `lh`. Received: ", kind)

        conf = []
        for desc, sob_vec in zip(self._params_desc, squere.T):
            p_type = desc['type']
            if p_type == 'category':
                idx = np.round(sob_vec*(len(desc['categories'])-1)).astype(int)
                val = [desc['categories'][i] for i in idx]
                conf.append(val)
            if p_type == 'float':
                low = desc['bounds'][0]
                high = desc['bounds'][1]
                val = low + sob_vec*(high-low)
                conf.append(val)
            if p_type == 'integer':
                low = desc['bounds'][0]
                high = desc['bounds'][1]
                val = (low + sob_vec*(high-low)).astype(int)
                conf.append(val)

        return pd.DataFrame(list(zip(*conf)), columns=[p['name'] for p in self._params_desc])
