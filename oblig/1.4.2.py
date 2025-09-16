import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import numpy as np


file_path = 'oblig\WineQT.csv' # you might have to change this to run locally
wine = pd.read_csv(file_path)
X = wine[['chlorides']]
y = wine['quality']

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", SGDRegressor())
])


skfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
r2 = cross_val_score(pipe, X, y, cv=skfolds, scoring='r2')
mse = cross_val_score(pipe, X, y, cv=skfolds, scoring='neg_mean_squared_error')
rmse = cross_val_score(pipe, X, y, cv=skfolds, scoring='neg_root_mean_squared_error')

print("Cross Validation Scores\n R2:", r2, "\n", "MSE:", mse, "\n", "RMSE:", rmse)

meanR = np.mean(r2) 
meanM = np.mean(mse) 
meanRM = np.mean(rmse)
variance = np.std(r2, ddof=1) 
varM = np.std(mse, ddof=1) 
varRM = np.std(rmse, ddof=1)
print("Mean result:" , "R^2", meanR, " MSE: ", meanM, " RMSE: ", meanRM )
print("Variance:", "R^2", variance, " MSE: ", varM, " RMSE: ", varRM )
