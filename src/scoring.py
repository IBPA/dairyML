from mutual_info import mutual_info_regression
from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, make_scorer
from skll.metrics import spearman, pearson

scoring = {'r2':make_scorer(r2_score), 
           'SRC':make_scorer(spearman), 
           'PCC':make_scorer(lambda x,y: pearson(x.flatten(),y.flatten())), 
           'MI':make_scorer(mutual_info_regression), 
           'MAE':make_scorer(mean_absolute_error)}