#!/usr/bin/python

print('Loading modules...')

import os, sys, getopt, datetime
import pickle as pkl
import pandas as pd
import numpy as np
from xgboost import XGBRegressor, XGBClassifier

from dairyml import XGBCombined

from skll.metrics import spearman, pearson
from sklearn.model_selection import cross_validate, RepeatedKFold
from sklearn.metrics import r2_score, mutual_info_score, make_scorer, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

import warnings
warnings.filterwarnings("ignore")

def main(argv):

	if len(argv) < 2:
		print('Not enough arguments specified\n Usage: test.py <model path> <input file path>')
		return
	else: 
		#Load the model
		model_path = argv[0]
		print('Loading model at {}'.format(model_path))
		with open(model_path, "rb" ) as f:
			model = pkl.load(f)

		#Load data
		data_path = argv[1]
		print('Loading data at {}'.format(data_path))
		data = pd.read_csv(data_path)
		data = data.set_index('FoodCode')
		numerical_features = data.columns[1:-1]

		# #shuffle??

		#Scale the features to 0 mean and unit variance
		print('Scaling input features...')
		ss = StandardScaler()
		X = pd.DataFrame(ss.fit_transform(data[numerical_features]),columns=data[numerical_features].columns,index=data.index)

		#Remove outliers
		print('Removing outliers...')
		iso = IsolationForest(contamination=.013)
		outliers = iso.fit_predict(X)
		print('Outliers removed: {}'.format(sum(outliers == -1)))
		X = X[outliers == 1]

		#Target variable
		Y = data['lac.per.100g'][outliers == 1]

		#splitter for CV
		splitter= RepeatedKFold(n_splits=10,n_repeats=5)

		#scoring
		scoring = {'r2':make_scorer(r2_score), 
		   'SRC':make_scorer(spearman), 
		   'PCC':make_scorer(pearson), 
		   'MI':make_scorer(mutual_info_score), 
		   'MAE':make_scorer(mean_absolute_error)}

		time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

		results_dir = '../reports/'
		results_path = '../reports/test_results_'+time+'.csv'

		if not os.path.exists(results_dir):
			os.makedirs(results_dir)

		print('Testing the model...')
		xgb_combined_results = cross_validate(model,X,Y,cv=splitter,scoring=scoring)

		print('\nResults: ')
		for score_name in scoring.keys():
			print('{}: {}'.format(score_name,np.round(np.mean(xgb_combined_results['test_'+score_name]),2)))
			overall_results.loc['XGB Combined',score_name] = np.round(np.mean(xgb_combined_results['test_'+score_name]),2)
		
		overall_results.to_csv(results_path)
		print('\nResults saved to {}'.format(results_path))
	return

if __name__ == "__main__":
	main(sys.argv[1:])
	