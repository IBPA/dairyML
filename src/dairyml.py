from __future__ import division

from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.base import BaseEstimator, RegressorMixin
import pandas as pd

import warnings
import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import _num_samples
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_consistent_length
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.random import random_choice_csc
from sklearn.utils.stats import _weighted_percentile
from sklearn.utils.multiclass import class_distribution

from xgboost import XGBRegressor, XGBClassifier


class XGBCombined(BaseEstimator,RegressorMixin):
    def __init__(self, max_depth_reg=None, max_depth_clas=None, importance_type='gain'):
        self.max_depth_reg = max_depth_reg
        self.max_depth_clas = max_depth_clas
        self.importance_type = importance_type

    def fit(self,X,y):
        self.reg = XGBRegressor(max_depth=self.max_depth_reg,colsample_bytree=0.9,importance_type=self.importance_type)
        self.clas = XGBClassifier(max_depth=self.max_depth_clas,importance_type=self.importance_type)
        self.reg.fit(X,y)
        y_binary = y != 0
        y_binary = y_binary.astype(int)
        self.clas.fit(X,y_binary)
        return self
        
    def predict(self, X):
        pred_reg = self.reg.predict(X)
        pred_clas = self.clas.predict(X)
        pred = np.multiply(pred_reg,pred_clas)
        return pred

class PerfectClassifierMeanRegressor():        
    def fit(self,X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y
        self.regressor = DummyRegressor(strategy='mean')
        
    def cross_val(self,scoring,k=10):
        self.scores = {}

        for name, scorer in scoring.items():
            for split in ['train','test']:
                self.scores[split+'_'+name] = []

        splitter = KFold(n_splits=k,shuffle=True,random_state=7)   
        for train_index, test_index in splitter.split(self.X,self.y):
            
            X_train = self.X.values[train_index]
            y_train = self.y.values[train_index]
            
            X_test = self.X.values[test_index]
            y_test = self.y.values[test_index]
            
            # get test y class labels for perfect classification
            y_test_binary = (y_test != 0)
            y_train_binary = (y_train != 0)

            self.regressor.fit(X_train,y_train.reshape(-1,1))
            
            reg_pred_test = self.regressor.predict(X_test).flatten()
            reg_pred_train = self.regressor.predict(X_train).flatten()

            y_pred_test = np.multiply(y_test_binary,reg_pred_test)
            y_pred_train = np.multiply(y_train_binary,reg_pred_train)

            for name, scorer in scoring.items():
                self.scores['test_'+name].append(scorer(y_test,y_pred_test))
                self.scores['train_'+name].append(scorer(y_train,y_pred_train))

                
        return self.scores
    
    def get_params(self):
        return(self.regressor.get_params())

        
def plot_r2(model,model_name,X,Y):
    y_pred = model.predict(X)
    plt.scatter(x=Y,y=y_pred,s=3)
    plt.xlabel('Actual value')
    plt.ylabel('Predicted value')
    plt.title('Actual vs. Predicted Values, {}'.format(model_name))
    
    
class BoundedLasso(BaseEstimator,RegressorMixin):
    def __init__(self, alpha=None):
        self.alpha = alpha
        
    def fit(self,X,y):
        self.lasso = Lasso(self.alpha)
        self.lasso.fit(X,y)
    
    def get_coef(self):
        return self.lasso.coef_
    
    def predict(self, x):
        pred_orig = self.lasso.predict(x)
        return np.clip(pred_orig,0,np.max(pred_orig))
        
class BoundedRidge(BaseEstimator,RegressorMixin):
    def __init__(self, alpha=None):
        self.alpha = alpha
        
    def fit(self,X,y):
        self.ridge = Ridge(self.alpha)
        self.ridge.fit(X,y)
    
    def get_coef(self):
        return self.ridge.coef_
    
    def predict(self, x):
        pred_orig = self.ridge.predict(x)
        return np.clip(pred_orig,0,np.max(pred_orig))
        
class BoundedLassoPlusLogReg(BaseEstimator,RegressorMixin):
    def __init__(self, alpha=None, C=None):
        self.alpha = alpha
        self.C = C

    def fit(self,X,y):
        self.reg = BoundedLasso(alpha=self.alpha)
        self.clas = LogisticRegression(penalty='l2',C=self.C,solver='lbfgs')
        self.reg.fit(X,y)
        y_binary = y != 0
        self.clas.fit(X,y_binary)
        return self
        
    def predict(self, X):
        pred_lasso = self.reg.predict(X)
        pred_logreg = self.clas.predict(X)
        pred = np.multiply(pred_lasso,pred_logreg)
        return pred

        
class BoundedRidgePlusLogReg(BaseEstimator,RegressorMixin):
    def __init__(self, alpha=None, C=None):
        self.alpha = alpha
        self.C = C

    def fit(self,X,y):
        self.reg = BoundedRidge(alpha=self.alpha)
        self.clas = LogisticRegression(penalty='l2',C=self.C,solver='lbfgs')
        self.reg.fit(X,y)
        y_binary = y != 0
        self.clas.fit(X,y_binary)
        return self
        
    def predict(self, X):
        pred_ridge = self.reg.predict(X)
        pred_logreg = self.clas.predict(X)
        pred = np.multiply(pred_ridge,pred_logreg)
        return pred     
        
def plot_coefficients(model,X):
    try:
        nonzero_coef_index = model.coef_ != 0
        coefficients = pd.DataFrame()
        coefficients['Feature'] = X.columns
        coefficients['coef'] = model.coef_
    except AttributeError:
        nonzero_coef_index = model.get_coef() != 0
        coefficients = pd.DataFrame()
        coefficients['Feature'] = X.columns
        coefficients['coef'] = model.get_coef()
    axs = coefficients[coefficients['coef']!=0].sort_values('coef').plot.barh(x='Feature',y='coef')
    axs.set_title('Feature Coefficients')
    axs.set_xlabel('Coefficient')
    
class DummyRegressorCustom(BaseEstimator, RegressorMixin):
    """
    DummyRegressor is a regressor that makes predictions using
    simple rules.
    This regressor is useful as a simple baseline to compare with other
    (real) regressors. Do not use it for real problems.
    Read more in the :ref:`User Guide <dummy_estimators>`.
    Parameters
    ----------
    strategy : str
        Strategy to use to generate predictions.
        * "mean": always predicts the mean of the training set
        * "median": always predicts the median of the training set
        * "quantile": always predicts a specified quantile of the training set,
          provided with the quantile parameter.
        * "constant": always predicts a constant value that is provided by
          the user.
    constant : int or float or array of shape = [n_outputs]
        The explicit constant as predicted by the "constant" strategy. This
        parameter is useful only for the "constant" strategy.
    quantile : float in [0.0, 1.0]
        The quantile to predict using the "quantile" strategy. A quantile of
        0.5 corresponds to the median, while 0.0 to the minimum and 1.0 to the
        maximum.
    Attributes
    ----------
    constant_ : float or array of shape [n_outputs]
        Mean or median or quantile of the training targets or constant value
        given by the user.
    n_outputs_ : int,
        Number of outputs.
    outputs_2d_ : bool,
        True if the output at fit is 2d, else false.
    """

    def __init__(self, strategy="mean", constant=None, quantile=None):
        self.strategy = strategy
        self.constant = constant
        self.quantile = quantile

    def fit(self, X, y, sample_weight=None):
        """Fit the random regressor.
        Parameters
        ----------
        X : {array-like, object with finite length or shape}
            Training data, requires length = n_samples
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            Target values.
        sample_weight : array-like of shape = [n_samples], optional
            Sample weights.
        Returns
        -------
        self : object
        """
        allowed_strategies = ("mean", "median", "quantile", "constant","median_nonzero")
        if self.strategy not in allowed_strategies:
            raise ValueError("Unknown strategy type: %s, expected one of %s."
                             % (self.strategy, allowed_strategies))

        y = check_array(y, ensure_2d=False)
        if len(y) == 0:
            raise ValueError("y must not be empty.")

        self.output_2d_ = y.ndim == 2
        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))
        self.n_outputs_ = y.shape[1]

        check_consistent_length(X, y, sample_weight)

        if self.strategy == "mean":
            self.constant_ = np.average(y, axis=0, weights=sample_weight)

        elif self.strategy == "median":
            if sample_weight is None:
                self.constant_ = np.median(y, axis=0)
            else:
                self.constant_ = [_weighted_percentile(y[:, k], sample_weight,
                                                       percentile=50.)
                                  for k in range(self.n_outputs_)]
        
        elif self.strategy == "median_nonzero":
            if sample_weight is None:
                self.constant_ = np.median(y[y > 0], axis=0)
            else:
                self.constant_ = [_weighted_percentile(y[y > 0][:, k], sample_weight,
                                                       percentile=50.)
                                  for k in range(self.n_outputs_)]

        elif self.strategy == "quantile":
            if self.quantile is None or not np.isscalar(self.quantile):
                raise ValueError("Quantile must be a scalar in the range "
                                 "[0.0, 1.0], but got %s." % self.quantile)

            percentile = self.quantile * 100.0
            if sample_weight is None:
                self.constant_ = np.percentile(y, axis=0, q=percentile)
            else:
                self.constant_ = [_weighted_percentile(y[:, k], sample_weight,
                                                       percentile=percentile)
                                  for k in range(self.n_outputs_)]

        elif self.strategy == "constant":
            if self.constant is None:
                raise TypeError("Constant target value has to be specified "
                                "when the constant strategy is used.")

            self.constant = check_array(self.constant,
                                        accept_sparse=['csr', 'csc', 'coo'],
                                        ensure_2d=False, ensure_min_samples=0)

            if self.output_2d_ and self.constant.shape[0] != y.shape[1]:
                raise ValueError(
                    "Constant target value should have "
                    "shape (%d, 1)." % y.shape[1])

            self.constant_ = self.constant

        self.constant_ = np.reshape(self.constant_, (1, -1))
        return self

    def predict(self, X, return_std=False):
        """
        Perform classification on test vectors X.
        Parameters
        ----------
        X : {array-like, object with finite length or shape}
            Training data, requires length = n_samples
        return_std : boolean, optional
            Whether to return the standard deviation of posterior prediction.
            All zeros in this case.
        Returns
        -------
        y : array, shape = [n_samples]  or [n_samples, n_outputs]
            Predicted target values for X.
        y_std : array, shape = [n_samples]  or [n_samples, n_outputs]
            Standard deviation of predictive distribution of query points.
        """
        check_is_fitted(self, "constant_")
        n_samples = _num_samples(X)

        y = np.full((n_samples, self.n_outputs_), self.constant_,
                    dtype=np.array(self.constant_).dtype)
        y_std = np.zeros((n_samples, self.n_outputs_))

        if self.n_outputs_ == 1 and not self.output_2d_:
            y = np.ravel(y)
            y_std = np.ravel(y_std)

        return (y, y_std) if return_std else y

    def score(self, X, y, sample_weight=None):
        """Returns the coefficient of determination R^2 of the prediction.
        The coefficient R^2 is defined as (1 - u/v), where u is the residual
        sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
        sum of squares ((y_true - y_true.mean()) ** 2).sum().
        The best possible score is 1.0 and it can be negative (because the
        model can be arbitrarily worse). A constant model that always
        predicts the expected value of y, disregarding the input features,
        would get a R^2 score of 0.0.
        Parameters
        ----------
        X : {array-like, None}
            Test samples with shape = (n_samples, n_features) or None.
            For some estimators this may be a
            precomputed kernel matrix instead, shape = (n_samples,
            n_samples_fitted], where n_samples_fitted is the number of
            samples used in the fitting for the estimator.
            Passing None as test samples gives the same result
            as passing real test samples, since DummyRegressor
            operates independently of the sampled observations.
        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True values for X.
        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.
        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.
        """
        if X is None:
            X = np.zeros(shape=(len(y), 1))
        return super().score(X, y, sample_weight)
        
def scores_to_df(df,model_name,scores,refit):
    cv_results = False
    for split in ['train','test']:
        for score_name in scores.keys():
            if 'mean_train_' in score_name or 'mean_test_' in score_name:
                df.loc[model_name,score_name[5:]] = np.round(scores[score_name][np.argmax(scores['mean_test_'+refit])],2)
                cv_results = True
        if not cv_results:
            for score_name in scores.keys():
                if 'train_' in score_name or 'test_' in score_name:
                    df.loc[model_name,score_name] = np.round(np.mean(scores[score_name]),2)
                    cv_results = True
    return df
