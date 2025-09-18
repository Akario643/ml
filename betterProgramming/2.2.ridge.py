from helperFunctions import wine, pipeline, coefficients, skfolds
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate

## select attributes
X = wine.drop(columns=['quality'])
y = wine['quality']

## create the model
pipe = pipeline(1, False, Ridge())
model = cross_validate(pipe, X, y, cv=skfolds ,return_estimator=True)
model_list = model['estimator']

## Get the coefficients
coef = coefficients(model_list, X)

