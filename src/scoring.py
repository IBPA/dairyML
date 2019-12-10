from mutual_info import mutual_info_regression
from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, make_scorer, accuracy_score, f1_score
from skll.metrics import spearman, pearson

scoring = {'r2':make_scorer(r2_score), 
           'SRC':make_scorer(spearman), 
           'PCC':make_scorer(lambda x,y: pearson(x.flatten(),y.flatten())), 
           'MI':make_scorer(lambda x,y: mutual_info_regression(x,y,random_state=7)), 
           'MAE':make_scorer(mean_absolute_error)}

scoring_test = {'r2':r2_score, 
           'SRC':spearman, 
           'PCC':pearson, 
           'MI': lambda x,y: mutual_info_regression(x.values.reshape(-1,1),y.values.reshape(-1,1),random_state=7)[0], 
           'MAE':mean_absolute_error}

scoring_clf = {'classifier_accuracy': accuracy_score, 
           'classifier_f1': f1_score }