from helperFunctions import wine, pipeline, scoring
from sklearn.tree import DecisionTreeRegressor

## Select Attributes
X = wine.drop(columns=['quality'])
y = wine['quality']

## Create model and show score
model = pipeline(2, True, DecisionTreeRegressor())
score = scoring(model, X, y, False)