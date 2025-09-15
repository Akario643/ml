import pandas as pd
from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

file_path = 'oblig\WineQT.csv' # you might have to change this to run locally
wine = pd.read_csv(file_path) 

X =  wine[['alcohol']]
y = wine['quality']

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", SGDRegressor())
])

model = pipe.fit(X,y)
prediction = model.predict(X)
plt.scatter(prediction, y)
plt.show()