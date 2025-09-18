import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_predict
import numpy as np
from sklearn.model_selection import cross_val_score


file_path = 'oblig\WineQT.csv' # you might have to change this to run locally
wine = pd.read_csv(file_path) 
wine = wine.drop(columns=['Id'])

X = wine.drop(columns=['quality'])
y = wine['quality']

skfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipe = Pipeline([
    ("preprocessor", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
    ("scale", StandardScaler()),
    ("model", SGDRegressor())
])

skfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_prediction = cross_val_predict(pipe, X,y, cv=skfolds)
plt.scatter(y, y_prediction, alpha=0.1)
plt.show()


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



