import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from ModelDetails import getmodelrdf
from Datatuning.DataSet import getDataset
from Regression import RegressionPipeLinear as r
from Datatuning.DataTuningROI import woNaN, woNaNOutliers, woOutliersMean

#ds = pd.read_csv(r"C:\Users\nicks\Desktop\python\Kaggle\Titanic Project")
dsmodels= getmodelrdf()
ds = getDataset()

i=1;
def calcAccuracyDS1(i):
    return round(r.regression(woNaN(ds), dsmodels['Model'].iloc[i], dsmodels['Parameter Grid'].iloc[i]),2)


def calcAccuracyDS2(i):
    return r.regression(woNaNOutliers(ds), dsmodels['Model'].iloc[i], dsmodels['Parameter Grid'].iloc[i])


def calcAccuracyDS3(i):
    return r.regression(woOutliersMean(ds), dsmodels['Model'].iloc[i], dsmodels['Parameter Grid'].iloc[i])


