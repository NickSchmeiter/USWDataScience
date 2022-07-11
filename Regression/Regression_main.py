import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from ModelDetails import getmodelrdf
from Datatuning.DataSet import getDataset
from Regression import RegressionPipeLinear as r
from Datatuning.DataTuningROI import woNaN, woNaNOutliers, woOutliersMean, woOutliersIQR

# gets df with model and grid param
dsmodels = getmodelrdf()
# gets dataset
ds = getDataset()


# calls regression methods for first ds and rounds
def calcAccuracyDS1(i):
    return round(r.regression(woNaN(ds), dsmodels['Model'].iloc[i], dsmodels['Parameter Grid'].iloc[i]),
                 4) * 100


# calls regression methods for second ds and rounds
def calcAccuracyDS2(i):
    return round(r.regression(woNaNOutliers(ds), dsmodels['Model'].iloc[i], dsmodels['Parameter Grid'].iloc[i]),
                 4) * 100


# calls regression methods for third ds and rounds
def calcAccuracyDS3(i):
    return round(r.regression(woOutliersMean(ds), dsmodels['Model'].iloc[i], dsmodels['Parameter Grid'].iloc[i]),
                 4) * 100


# calls regression methods for fourth ds and rounds
def calcAccuracyDS4(i):
    return round(r.regression(woOutliersIQR(ds), dsmodels['Model'].iloc[i], dsmodels['Parameter Grid'].iloc[i]),
                 4) * 100
