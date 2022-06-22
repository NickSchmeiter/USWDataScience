import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from DataSet import getDataset
from Regression_Pipe_Linear import regression
from DataTuningROI import woNaNOutliers, woOutliersMean

ds = getDataset()

regression(ds)
regression(woNaNOutliers(ds))
regression(woOutliersMean(ds))

