import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import numpy as np

filepath = 'oblig/WineQT.csv'
wine = pd.read_csv(filepath)
wine = wine.drop(columns=['Id'])

skfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

"""
Creates a pipeline which can use whatever model the user wishes
@poly - do you want polynomial features in the model (1) for no all other values are the degree you want
@interaction_terms - do you want relationships between variables (True/False)
@model - the model you want to use
"""
def pipeline(poly, interaction_terms ,model):
    pipe = Pipeline([
            ("preprocessor", PolynomialFeatures(degree=poly, interaction_only=interaction_terms, include_bias=False)),
            ("scale", StandardScaler()),
            ("model", model)
            ])
    return pipe


"""
prints out the cross validation scores from a model
@model - the model to evaluate
@x - predictor values
@y - what to predict
@folds - train-test split
"""
def scoring(model,x,y, folds):
    r2 = cross_val_score(model, x, y, cv=skfolds, scoring='r2')
    mse = cross_val_score(model, x, y, cv=skfolds, scoring='neg_mean_squared_error')
    rmse = cross_val_score(model, x, y, cv=skfolds, scoring='neg_root_mean_squared_error')

    meanR = np.mean(r2) 
    meanM = np.mean(mse) 
    meanRM = np.mean(rmse)
    variance = np.std(r2, ddof=1) 
    varM = np.std(mse, ddof=1) 
    varRM = np.std(rmse, ddof=1)

    if(folds==True):
        print("Cross Validation Scores\n R2:", r2, "\n", "MSE:", -mse, "\n", "RMSE:", -rmse)
        print("\n")
    else:
        print("Mean result:" , "R^2", meanR, " MSE: ", -meanM, " RMSE: ", -meanRM )
        print("Variance:", "R^2", variance, " MSE: ", varM, " RMSE: ", varRM )

    return 0

"""
Creates a coefficient dataframe which stores collumn names together with coefficients
@model - model to get coefficients from
@x - predictor values
"""
def coefficients(model, x):
    ## Create a dictionary of coefficients
    coef_list = [] ## create an empty list
    for i in range(len(model)): ## loop through all folds
        coef_list.append(model[i].named_steps['model'].coef_) ## add eachs fold's model to the coefficient list
    coef_list = np.vstack(coef_list) ## vertically stack the list (make the list 1D)
    coef_list = np.mean(coef_list, axis = 0) ## take the mean of the coefficients
    ## create the dictionary/dataframe of coefficients
    coef_table = pd.DataFrame(list(x.columns)).copy()
    coef_table.insert(len(coef_table.columns), "Coefs", coef_list)
    print(coef_table)




