import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression

X = './Risk_Diabetes/X_res_bf.pkl'
Y = './Risk_Diabetes/y_res_bf.pkl'

LR = LogisticRegression(C=0.5, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, max_iter=500,
                   multi_class='auto', n_jobs=-1, penalty='l2', random_state=20,
                   solver='lbfgs', tol=0.0001, verbose=0, warm_start=False)

with open(X, 'rb') as file:
    X_res = pickle.load(file)

with open(Y, 'rb') as file:
    y_res = pickle.load(file)

LR.fit(X_res, y_res)
    
def Model_One(data):
  risk = LR.predict_proba(data.reshape(1,-1))
  v = round(risk[0][0],3)
  return v

def Model_Batch(data):
  risk = LR.predict_proba(data)
  return risk[:,0]


