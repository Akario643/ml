from helperFunctions import wine, pipeline
from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as plt

## Select the X and Y attributes
X = wine[['chlorides']]
y = wine['quality']

## Create model
model = pipeline(1,False,SGDRegressor()).fit(X,y)
prediction = model.predict(X)

## Create the plot
plt.scatter(X,y)
plt.plot(X, prediction, color='red')
plt.xlabel("Chlorides")
plt.ylabel('Quality')
plt.show()