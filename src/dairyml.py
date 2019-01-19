from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.base import BaseEstimator, RegressorMixin
import pandas as pd

class PerfectClassifierMeanRegressor():        
    def fit(self,X,y):
        self.X = X
        self.y = y
        self.regressor = DummyRegressor(strategy='mean')
        
    def cross_val(self,scoring,k=10):
        self.scores = {}
        splitter = KFold(n_splits=k,shuffle=True,random_state=7)   
        for train_index, test_index in splitter.split(self.X,self.y):
            
            X_train = self.X.values[train_index]
            y_train = self.y.values[train_index]
            
            X_test = self.X.values[test_index]
            y_test = self.y.values[test_index]
            
            # get test y class labels for perfect classification
            y_test_binary = y_test != 0

            self.regressor.fit(X_train,y_train)
            
            reg_pred = self.regressor.predict(X_test)
            
            y_pred = np.multiply(y_test_binary,reg_pred)
            for name, scorer in scoring.items():
                print(scorer)
                try:
                    self.scores[name].append(scorer(y_true=y_test,y_pred=y_pred))
                except KeyError:
                    self.scores[name] = scorer(y_true=y_test,y_pred=y_pred)
            
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
		
class BoundedLassoPlusLogReg(BaseEstimator,RegressorMixin):
    def __init__(self, alpha=None, C=None):
        self.alpha = alpha
        self.C = C

    def fit(self,X,y):
        self.bounded_lasso = BoundedLasso(alpha=self.alpha)
        self.logreg = LogisticRegression(penalty='l2',C=self.C,solver='lbfgs')
        self.bounded_lasso.fit(X,y)
        y_binary = y != 0
        self.logreg.fit(X,y_binary)
        return self
        
    def predict(self, X):
        pred_lasso = self.bounded_lasso.predict(X)
        pred_logreg = self.logreg.predict(X)
        pred = np.multiply(pred_lasso,pred_logreg)
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
		