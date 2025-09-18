from helperFunctions import wine, pipeline, scoring, skfolds
from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict

## select attributes
X = wine.drop(columns=['quality'])
y = wine['quality']

model = pipeline(2, True, SGDRegressor())
score = scoring(model, X, y, False)

y_prediction = cross_val_predict(model, X, y, cv=skfolds)
plt.scatter(y, y_prediction, alpha=0.1)
plt.xlabel("Actual Values")
plt.ylabel("Predicted values")
plt.show()
