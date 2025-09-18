import pandas as pd
from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

file_path = 'oblig\WineQT.csv' # you might have to change this to run locally
wine = pd.read_csv(file_path) 

## Select attributes to use
X1 =  wine[['alcohol']]
X2 = wine[['chlorides']]
y = wine['quality']

## Create pipeline and model
pipe1 = Pipeline([
    ("scale", StandardScaler()),
    ("model", SGDRegressor())
])

pipe2 = Pipeline([
    ("scale", StandardScaler()),
    ("model", SGDRegressor())
])
modelAlc = pipe1.fit(X1,y)
modelChlor = pipe2.fit(X2,y)

## Print out coefficient data
print("Coefficient Alcohol",modelAlc.named_steps['model'].coef_)
print("Intercept Alcohol:", modelAlc.named_steps['model'].intercept_)
print("\nCoefficient Chlorides:", modelChlor.named_steps['model'].coef_)
print("Intercept Chlorides", modelChlor.named_steps['model'].intercept_)