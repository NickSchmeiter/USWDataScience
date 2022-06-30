import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from ModelDetails import getmodelrdf
from Datatuning.DataSet import getDataset
from RegressionPipeLinear import regression
from Datatuning.DataTuningROI import woNaN, woNaNOutliers, woOutliersMean

dsmodels= getmodelrdf()
ds = getDataset()


def calcAccuracyDS1(i):
    return regression(woNaN(ds), dsmodels['Model'].iloc[i], dsmodels['Parameter Grid'].iloc[i])


def calcAccuracyDS2(i):
    return regression(woNaNOutliers(ds), dsmodels['Model'].iloc[i], dsmodels['Parameter Grid'].iloc[i])


def calcAccuracyDS3(i):
    return regression(woOutliersMean(ds), dsmodels['Model'].iloc[i], dsmodels['Parameter Grid'].iloc[i])

