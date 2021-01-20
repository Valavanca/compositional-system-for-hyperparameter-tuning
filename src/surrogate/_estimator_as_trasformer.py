from sklearn.base import TransformerMixin, BaseEstimator, MetaEstimatorMixin
from sklearn import clone
from sklearn.utils.validation import check_is_fitted, check_X_y


class HypothesisTransformer(TransformerMixin, MetaEstimatorMixin, BaseEstimator):
    """ Allows using an estimator such as a transformator in an earlier step of a pipeline.

        Args:
            estimator (BaseEstimator): An instance of the estimator that should be used for the transformation.
            predict_func (str, optional): Function of the estimator that will be used as transformer. Defaults to 'predict'.

        Ref:
            scikit-lego: https://github.com/koaning/scikit-lego
    """
    def __init__(self, estimator: BaseEstimator, predict_func: str = 'predict'):
        self.estimator = estimator
        self.estimator_ = None
        self.predict_func = predict_func

    def fit(self, X, y):
        """ Fits the estimator.

        Args:
            X (array-like, sparse matrix of shape (n_samples, n_features)): The input samples.
            y ([array-like of shape (n_samples,)): Target values or labels

        Returns:
            self: object
        """        
        X, y = check_X_y(X, y, estimator=self)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """ Fits and returns estimator. The final fitted estimator is evaluated as 
        transformed data for the following step in the pipeline.  

        Args:
            X (array-like, sparse matrix of shape (n_samples, n_features)): The input samples.
            y ([array-like of shape (n_samples,)): Target values or labels

        Returns:
            [type]: [description]
        """

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
