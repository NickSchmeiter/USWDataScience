from sklearn.linear_model import LinearRegression

from DataSet import getDataset
from Classification_Pipe_Linear import classification
from DataTuningROI import woNaNOutliers, woOutliersMean,woNaN


model_linear_regression=LinearRegression()
grid_param_linear_regression = {
    'fit_intercept': [True, False],
    'copy_X': [True, False],
    'normalize': [True, False],
    'positive': [True, False]
}
ds = getDataset()

classification(woNaN(ds),model_linear_regression,grid_param_linear_regression)
classification(woNaNOutliers(ds),model_linear_regression,grid_param_linear_regression)
classification(woOutliersMean(ds),model_linear_regression,grid_param_linear_regression)

