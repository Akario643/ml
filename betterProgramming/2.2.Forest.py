from helperFunctions import wine, pipeline, scoring, skfolds
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

## Select Attributes
X = wine.drop(columns=['quality'])
y = wine['quality']

## Create model and show score
pipe = pipeline(1, False, RandomForestRegressor())
parameters = {'model__n_estimators':[100,200,400,500], 'model__max_depth':[2,5,8,10,20], "model__min_samples_split":[2,5,10,15], "model__min_samples_leaf":[1,3,4,7]}
grid = GridSearchCV(pipe, parameters, cv=skfolds)
grid.fit(X,y)
model_grid = grid.best_estimator_

score = scoring(model_grid, X,y,False)
