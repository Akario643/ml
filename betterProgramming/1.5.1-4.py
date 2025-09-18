from helperFunctions import wine, pipeline, scoring, skfolds
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt

## Select all attributes
X = wine.drop(columns=['quality'])
y = wine['quality']

## Create model
model = pipeline(1, False, SGDRegressor())

## scoring
score = scoring(model, X, y, False)

## using model to predict
y_prediction = cross_val_predict(model, X,y, cv=skfolds)


## plot the data
plt.scatter(y,y_prediction, alpha=0.1)
plt.xlabel("Actual Values")
plt.ylabel("Predicted values")
plt.show()