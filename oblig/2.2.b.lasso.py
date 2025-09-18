from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
import numpy as np
from sklearn.model_selection import GridSearchCV

filepath = 'oblig/WineQT.csv'
wine = pd.read_csv(filepath)
wine = wine.drop(columns=['Id'])

X = wine.drop(columns=['quality'])
y = wine['quality']

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", Lasso())
])

#80 20 split
skfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

## find out what the best parameter for regularization strength
parameters = {'model__alpha':[0.001,0.01,0.1,1]} ## different alpha levels to try in lasso
grid = GridSearchCV(pipe,parameters, cv=skfolds) ## try the different levels
grid.fit(X,y) ## fit the model
model_grid = grid.best_estimator_ ## get the best model 


### create the model
model = cross_validate(model_grid, X, y, cv=skfolds, return_estimator=True)
model_list = model['estimator']


### list coefficiants in a table
coef_list = [] 
for i in range(len(model_list)):
    coef_list.append(model_list[i].named_steps['model'].coef_)
    
coef_list = np.vstack(coef_list) # create one big array
coef_list = np.mean(coef_list, axis = 0)
coef_table = pd.DataFrame(list(X.columns)).copy()
coef_table.insert(len(coef_table.columns), "Coefs", coef_list)
print(coef_table)
