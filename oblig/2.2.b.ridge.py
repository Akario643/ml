from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
import numpy as np
from sklearn.model_selection import GridSearchCV

### Read file and set X and y values
filepath = 'oblig/WineQT.csv' # if it doesn't work, try to change this
wine = pd.read_csv(filepath)
wine = wine.drop(columns=['Id'])
X = wine.drop(columns=['quality'])
y = wine['quality']

### model
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", Ridge())
])

skfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = cross_validate(pipe, X, y, cv=skfolds, return_estimator=True)
model_list = model['estimator']


### list coefficiants in a table
coef_list = [] # empty list
for i in range(len(model_list)):
    coef_list.append(model_list[i].named_steps['model'].coef_) ## add coefficients to the list
    
coef_list = np.vstack(coef_list) # vertically stack the array (make the array 1D)
coef_list = np.mean(coef_list, axis = 0) ## take the mean of all coefficients
coef_table = pd.DataFrame(list(X.columns)).copy()
coef_table.insert(len(coef_table.columns), "Coefs", coef_list)
print(coef_table)

