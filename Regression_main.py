import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from DataSet import getDataset
from Regression_Pipe_Linear import regression
from DataTuningROI import woNaNOutliers, woOutliersMean, woNaN
from sklearn.linear_model import LinearRegression


model_linear_regression=LinearRegression()
grid_param_linear_regression = {
    'fit_intercept': [True, False],
    'copy_X': [True, False],
    'normalize': [True, False],
    'positive': [True, False]
}
ds = getDataset()

regression(woNaN(ds),model_linear_regression,grid_param_linear_regression)
regression(woNaNOutliers(ds),model_linear_regression,grid_param_linear_regression)
regression(woOutliersMean(ds),model_linear_regression,grid_param_linear_regression)

