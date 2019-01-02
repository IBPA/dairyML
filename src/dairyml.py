from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt

class PerfectClassifierMeanRegressor():        
    def fit(self,X,y):
        self.X = X
        self.y = y
        self.regressor = DummyRegressor(strategy='mean')
        
    def cross_val(self,k=10):
        self.scores = []
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
            
            self.scores.append(r2_score(y_true=y_test,y_pred=y_pred))
            
        return self.scores

		
def plot_r2(model,model_name,X,Y):
    y_pred = model.predict(X)
    plt.scatter(x=Y,y=y_pred,s=3)
    plt.xlabel('Actual value')
    plt.ylabel('Predicted value')
    plt.title('Actual vs. Predicted Values, {}'.format(model_name))
		