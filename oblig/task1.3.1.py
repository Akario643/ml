import pandas as pd
from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

file_path = 'oblig\WineQT.csv' # you might have to change this to run locally
wine = pd.read_csv(file_path) 

X =  wine[['chlorides']] # select chlorides as attribute X
y = wine['quality'] # select quality as attribute y

## Create the model and pipeline
pipe = Pipeline([
    ("scale", StandardScaler()), ## scale the attributes
    ("model", SGDRegressor()) ## use the SGDRegressor 
])

model = pipe.fit(X,y) ## fit the model
prediction = model.predict(X) ## predict the value

## create the plot
plt.scatter(X, y)
plt.plot(X, prediction, color="red")
plt.xlabel("Chlorides")
plt.ylabel("Quality")
plt.show()