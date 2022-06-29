from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
def getmodeldf():

    model_linear_regression=LinearRegression()
    grid_param_linear_regression = {
    'fit_intercept': [True, False],
    'copy_X': [True, False],
    'normalize': [True, False],
    'positive': [True, False]
}
    tree_model= DecisionTreeClassifier()
    grid_param_tree = {
    'min_samples_split': [2, 10,100],
    'splitter': ['best', 'random'],
    'max_features': ['auto', 'sqrt','log']
}
    forest_model= RandomForestClassifier()
    grid_param_forest = {
        'min_samples_split': [2, 10, 100],
        'max_features': ['auto', 'sqrt', 'log2'],
        'bootstrap': [True,False],
        'n_estimators': [10,100,1000],
        'max_depth': [10, 30, 50, 70, 90, 100]
    }

    data = [[model_linear_regression, grid_param_linear_regression], [tree_model, grid_param_tree],[forest_model,grid_param_forest]]
    df = pd.DataFrame(data, columns=['Model', 'Parameter Grid'])
    return df