#!/usr/bin/python

print('Loading modules...')

import os, sys, getopt, datetime
import pickle as pkl
import pandas as pd
import numpy as np

from xgboost import XGBRegressor, XGBClassifier
from dairyml import XGBCombined

from skll.metrics import spearman, pearson
from sklearn.utils import shuffle
from sklearn.model_selection import cross_validate, RepeatedKFold
from sklearn.metrics import r2_score, make_scorer, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from scoring import *



import warnings
warnings.filterwarnings("ignore")

results_dir = '../reports/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

def my_load_model(model_path):
    print('Loading model at {}'.format(model_path))
    if 'ffnn' in model_path:
        from tensorflow.keras.models import load_model
        model = load_model(model_path)
    else:
        with open(model_path, "rb" ) as f:
            model = pkl.load(f)

    return model


def scale_data(data):
    print('Scaling input features...')
    train_means = np.loadtxt('./scaling/train_feature_means.csv',delimiter=',')
    train_vars = np.loadtxt('./scaling/train_feature_variances.csv',delimiter=',')
    scaled_data = (data - train_means) / np.sqrt(train_vars)

    return scaled_data

def get_model_predictions(model,X):
    # Get model predictions
    print('Testing the model... ')
    # full predictions
    predictions = pd.DataFrame(index=X.index)

    Y_pred = model.predict(X)

    predictions['Y_pred'] = Y_pred
    predictions['Y_pred_reg'] = np.nan
    predictions['Y_pred_clas'] = np.nan
    
    try:
        Y_pred_reg = model.reg.predict(X) # regressor predictions (combined models only)
        predictions['Y_pred_reg'] = Y_pred_reg

        Y_pred_clas = model.clas.predict(X) #classifier predictions (combined models only)
        predictions['Y_pred_clas'] = Y_pred_clas
        predictions['Y_pred_clas'] = predictions['Y_pred_clas'].astype(int)

    except AttributeError:
        pass

    return predictions

def score_predictions(predictions,model_name):
    results = pd.DataFrame()
    #score the predictions
    print('\nResults: ')
    for name, metric in scoring_test.items():
        score = np.round(metric(predictions['Y'],predictions['Y_pred']),2)
        print('{}: {}'.format(name,score))
        results.loc[model_name,name] = score

    if not all(np.isnan(predictions['Y_pred_clas'])):
        for name, metric in scoring_clf.items():
            score = np.round(metric(predictions['Y_binary'],predictions['Y_pred_clas']),2)
            print('{}: {}'.format(name,score))
            results.loc[model_name,name] = score
    else:
        for name, metric in scoring_clf.items():
            score = np.nan
            print('{}: {}'.format(name,score))
            results.loc[model_name,name] = score

    return results


def main(argv):

    if len(argv) < 2:
        print('Not enough arguments specified\n Usage: test.py <model path> <input file path>')
        return
    else: 
        #Load the model, pretrained on the full training data
        model_path = argv[0]
        model = my_load_model(model_path)

        #Load test data
        data_path = argv[1]
        print('Loading data at {}'.format(data_path))
        data = pd.read_csv(data_path)
        data = data.set_index('FoodCode')
        numerical_features = data.columns[1:-1]

        # scale the data
        X = scale_data(data[numerical_features])

        # get the target variable
        Y = data['lac.per.100g']

        # get model predictions
        predictions = get_model_predictions(model,X)
        predictions['Y'] = Y #actual values
        predictions['Y_binary'] = (Y != 0)

        # score the predictions
        model_name = os.path.basename(model_path).replace('.model','')
        results = score_predictions(predictions,model_name)

        #store the results
        time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        results_path = 'reports/test_results_'+time+'.csv'
        predictions_path = 'reports/test_predictions_'+time+'.csv'

        results.to_csv(results_path)
        predictions.to_csv(predictions_path)

        print('\nResults saved to {}'.format(results_path))
        print('\nPredictions saved to {}'.format(predictions_path))

        return

if __name__ == "__main__":
    main(sys.argv[1:])
    