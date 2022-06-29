import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from ModelDetails import getmodelrdf
from DataSet import getDataset
from Regression_Pipe_Linear import regression
from DataTuningROI import woNaNOutliers, woOutliersMean, woNaN

dsmodels= getmodelrdf()

ds = getDataset()

regression(woNaN(ds),dsmodels['Model'].iloc[1]
,dsmodels['Parameter Grid'].iloc[1])
regression(woNaNOutliers(ds),dsmodels['Model'].iloc[1]
,dsmodels['Parameter Grid'].iloc[1])
regression(woOutliersMean(ds),dsmodels['Model'].iloc[1]
,dsmodels['Parameter Grid'].iloc[1])

