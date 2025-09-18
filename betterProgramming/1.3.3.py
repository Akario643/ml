from helperFunctions import wine, pipeline
from sklearn.linear_model import SGDRegressor

## Select Attributes
X1 = wine[['alcohol']]
X2 = wine[['chlorides']]
y = wine['quality']

## Create models
modelChlor = pipeline(1,False, SGDRegressor()).fit(X2,y)
modelAlc = pipeline(1, False, SGDRegressor()).fit(X1,y)

## print out coefficiants
print("Coefficient Alcohol",modelAlc.named_steps['model'].coef_)
print("Intercept Alcohol:", modelAlc.named_steps['model'].intercept_)
print("\nCoefficient Chlorides:", modelChlor.named_steps['model'].coef_)
print("Intercept Chlorides", modelChlor.named_steps['model'].intercept_)


