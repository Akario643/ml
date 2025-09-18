import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
import numpy as np

file_path = 'oblig\WineQT.csv' # you might have to change this to run locally
wine = pd.read_csv(file_path) 
wine = wine.drop(columns=['Id'])

## Select attributes
X = wine.drop(columns=['quality'])
y = wine['quality']

## Create pipeline
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", SGDRegressor())
])

## Create train test split
skfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

## Create model
model = cross_validate(pipe, X, y, cv=skfolds, return_estimator=True)
model_list = model['estimator'] ## Get each fold's model

## Create a dictionary of coefficients
coef_list = [] ## create an empty list
for i in range(len(model_list)): ## loop through all folds
    coef_list.append(model_list[i].named_steps['model'].coef_) ## add eachs fold's model to the coefficient list
    
coef_list = np.vstack(coef_list) ## vertically stack the list (make the list 1D)
coef_list = np.mean(coef_list, axis = 0) ## take the mean of the coefficients
## create the dictionary/dataframe of coefficients
coef_table = pd.DataFrame(list(X.columns)).copy()
coef_table.insert(len(coef_table.columns), "Coefs", coef_list)
print(coef_table)