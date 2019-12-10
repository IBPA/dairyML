from sklearn.model_selection import KFold, RepeatedKFold


ten_fold_5_repeat = RepeatedKFold(n_splits=10,n_repeats=5,random_state=7)

ten_fold_no_repeat = KFold(n_splits=10,shuffle=True,random_state=7)

five_fold_no_repeat = KFold(n_splits=5,shuffle=True,random_state=7)