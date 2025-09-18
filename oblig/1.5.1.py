import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy as np

file_path = 'oblig\WineQT.csv' # you might have to change this to run locally
wine = pd.read_csv(file_path) 
wine = wine.drop(columns=['Id'])

## select the attributes
X = wine.drop(columns=['quality'])
y = wine['quality']

skfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) ## create train test split

## create the pipeline
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", SGDRegressor())
])

## Cross validation scores
r2 = cross_val_score(pipe, X, y, cv=skfolds, scoring='r2')
mse = cross_val_score(pipe, X, y, cv=skfolds, scoring='neg_mean_squared_error')
rmse = cross_val_score(pipe, X, y, cv=skfolds, scoring='neg_root_mean_squared_error')

meanR = np.mean(r2) 
meanM = np.mean(mse) 
meanRM = np.mean(rmse)
variance = np.std(r2, ddof=1) 
varM = np.std(mse, ddof=1) 
varRM = np.std(rmse, ddof=1)
print("Mean result:" , "R^2", meanR, " MSE: ", meanM, " RMSE: ", meanRM )
print("Variance:", "R^2", variance, " MSE: ", varM, " RMSE: ", varRM )