import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression

X = './Risk_Diabetes/X_res.pkl'
Y = './Risk_Diabetes/y_res.pkl'

LR = LogisticRegression(C=0.5, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=15, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)

with open(X, 'rb') as file:
    X_res = pickle.load(file)

with open(Y, 'rb') as file:
    y_res = pickle.load(file)

LR.fit(X_res, y_res)
