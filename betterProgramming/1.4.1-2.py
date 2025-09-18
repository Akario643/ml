from helperFunctions import wine, pipeline, scoring
from sklearn.linear_model import SGDRegressor

## Select Attributes
X1 = wine[['alcohol']]
X2 = wine[['chlorides']]
y = wine['quality']

## Create models
modelChlor = pipeline(1,False, SGDRegressor())
modelAlc = pipeline(1, False, SGDRegressor())

## Print out the scores
chlorScore = scoring(modelChlor, X2, y, True)
alcScore = scoring(modelAlc, X1, y, True)