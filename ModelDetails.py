from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, \
    GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import pandas as pd
def getmodelrdf():

    model_linear_regression=LinearRegression()
    grid_param_linear_regression = {
    'fit_intercept': [True, False],
    'copy_X': [True, False],
    'normalize': [True, False],
    'positive': [True, False]
}
    tree_model= DecisionTreeRegressor()
    grid_param_tree = {
    'min_samples_split': [2, 10,20,50],
    'splitter': ['best', 'random'],
    'max_features': ['auto', 'sqrt','log2']
}
    forest_model= RandomForestRegressor()
    grid_param_forest = {
        'min_samples_split': [2, 10, 20, 100],
        'max_features': ['auto', 'sqrt', 'log2'],
        'bootstrap': [True, False],
        'n_estimators': [10, 100, 200],
        'max_depth': [10, 30, 50, 70, 90, 100]
    }
    svc_regression = LinearSVR()
    grid_param_svc_regression = {
        'C': [0.1, 1, 10],
        'max_iter': [1000, 10000, 100000]
    }
    gradient_boost = GradientBoostingRegressor()
    grid_param_boost = {
        'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
         'max_features' : ['auto', 'sqrt', 'log2'],
        'n_estimators': [10, 100, 200],
        'min_samples_split': [2, 10, 20, 50]
    }
    data = [[model_linear_regression, grid_param_linear_regression], [tree_model, grid_param_tree], [forest_model, grid_param_forest], [svc_regression,grid_param_svc_regression], [gradient_boost, grid_param_boost]]
    df = pd.DataFrame(data, columns=['Model', 'Parameter Grid'])
    return df

def getmodelcdf():

    model_linear_regression=LinearRegression()
    grid_param_linear_regression = {
    'fit_intercept': [True, False],
    'copy_X': [True, False],
    'normalize': [True, False],
    'positive': [True, False]
}
    tree_model= DecisionTreeClassifier()
    grid_param_tree = {
    'min_samples_split': [2, 10,20,50],
    'splitter': ['best', 'random'],
    'max_features': ['auto', 'sqrt','log2']
}
    forest_model= RandomForestClassifier()
    grid_param_forest = {
        'min_samples_split': [2, 10,20, 100],
        'max_features': ['auto', 'sqrt', 'log2'],
        'bootstrap': [True,False],
        'n_estimators': [10,100,200],
        'max_depth': [10, 30, 50, 70, 90, 100]
    }
    svc_classification=LinearSVC()
    grid_param_svc_classification = {
        'C': [0.1, 1, 10],
        'max_iter': [1000, 10000, 100000]
    }
    gradient_boost = GradientBoostingClassifier()
    grid_param_boost = {
        'max_features': ['auto', 'sqrt', 'log2'],
        'n_estimators': [10, 100, 200],
        'min_samples_split': [2, 10, 20, 50]
    }
    data = [[model_linear_regression, grid_param_linear_regression], [tree_model, grid_param_tree],[forest_model,grid_param_forest],[svc_classification,grid_param_svc_classification],[gradient_boost,grid_param_boost]]
    df = pd.DataFrame(data, columns=['Model', 'Parameter Grid'])
    return df