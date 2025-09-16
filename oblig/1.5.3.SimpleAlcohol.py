import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
import numpy as np


file_path = 'oblig\WineQT.csv' # you might have to change this to run locally
wine = pd.read_csv(file_path)
X = wine[['alcohol']]
y = wine['quality']

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", SGDRegressor())
])


skfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_prediction = cross_val_predict(pipe, X,y, cv=skfolds)
plt.scatter(y, y_prediction, alpha=0.1)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red", linestyle='dashed')
plt.xlabel("Actual values")
plt.ylabel("Prediction")
plt.title("Alcohol Simple regression")
plt.show()
