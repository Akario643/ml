from helperFunctions import wine, pipeline, scoring
from sklearn.ensemble import RandomForestRegressor

## Select Attributes
X = wine.drop(columns=['quality'])
y = wine['quality']

## Create model and show score
model = pipeline(2, True, RandomForestRegressor())
score = scoring(model, X, y, False)