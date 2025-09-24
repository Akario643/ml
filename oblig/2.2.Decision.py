from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.model_selection import cross_validate
import numpy as np
from sklearn.model_selection import GridSearchCV

filepath = 'oblig/WineQT.csv' # if it doesn't work, try to change this
wine = pd.read_csv(filepath)
wine = wine.drop(columns=['Id'])

## Select attributes
X = wine.drop(columns=['quality'])
y = wine['quality']

## create pipeline
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", DecisionTreeRegressor())
])

## Cross validation
skfolds  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) ## create train test split

## find out what the best parameters for the decision tree
parameters = {'model__max_depth':[2,5,8,10,None], "model__min_samples_leaf":[1,2,4,6], "model__min_samples_split":[2,5,10,15]}
grid = GridSearchCV(pipe,parameters, cv=skfolds) ## try the different levels
grid.fit(X,y) ## fit the model
model_grid = grid.best_estimator_ ## get the best model 


#cross validate the model
r2 = cross_val_score(model_grid, X, y, cv=skfolds, scoring='r2')
mse = cross_val_score(model_grid, X, y, cv=skfolds, scoring='neg_mean_squared_error')
rmse = cross_val_score(model_grid, X, y, cv=skfolds, scoring='neg_root_mean_squared_error')


meanR = np.mean(r2) 
meanM = np.mean(mse) 
meanRM = np.mean(rmse)
variance = np.std(r2, ddof=1) 
varM = np.std(mse, ddof=1) 
varRM = np.std(rmse, ddof=1)
print("Mean result:" , "R^2", meanR, " MSE: ", meanM, " RMSE: ", meanRM )
print("Variance:", "R^2", variance, " MSE: ", varM, " RMSE: ", varRM )
