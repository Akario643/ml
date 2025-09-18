from helperFunctions import wine, pipeline, coefficients, skfolds
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_validate, GridSearchCV

## select attributes
X = wine.drop(columns=['quality'])
y = wine['quality']

## Find out penalty rate and create pipe
pipe = pipeline(1, False, Lasso())
parameters = {'model__alpha':[0.001, 0.01, 0.05, 0.1, 1]}
grid = GridSearchCV(pipe, parameters, cv=skfolds)
grid.fit(X,y)
model_grid = grid.best_estimator_


## Create model
model = cross_validate(model_grid, X, y, cv=skfolds, return_estimator=True)
model_list = model['estimator']

## Get the coefficients
coef = coefficients(model_list, X)

