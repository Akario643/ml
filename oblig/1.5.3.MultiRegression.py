import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
import numpy as np

file_path = 'oblig\WineQT.csv' # you might have to change this to run locally
wine = pd.read_csv(file_path) 
wine = wine.drop(columns=['Id'])

X = wine.drop(columns=['quality'])
y = wine['quality']

skfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", LinearRegression())
])

y_prediction = cross_val_predict(pipe, X,y, cv=skfolds)
plt.scatter(y, y_prediction, alpha=0.1)
plt.xlabel("Actual values")
plt.ylabel("Prediction")
plt.title("Multi Regression")
plt.show()