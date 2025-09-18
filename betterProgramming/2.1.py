from helperFunctions import wine, pipeline, skfolds, coefficients
from sklearn.model_selection import cross_validate
from sklearn.linear_model import SGDRegressor

## select attributes
X = wine.drop(columns=['quality'])
y = wine['quality']

## create model
pipe = pipeline(1, False, SGDRegressor())
model = cross_validate(pipe, X, y, cv=skfolds ,return_estimator=True)
model_list = model['estimator']

## Get coefficients
coef = coefficients(model_list, X)
