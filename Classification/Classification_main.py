from sklearn.linear_model import LinearRegression

from Datatuning.DataSet import getDataset
from Classification_Pipe_Linear import classification
from Datatuning.DataTuningROI import woNaNOutliers, woOutliersMean,woNaN
from ModelDetails import getmodelcdf

dsmodels= getmodelcdf()

ds = getDataset()
def calcAccuracyDS1(i):
    return classification(woNaN(ds),dsmodels['Model'].iloc[i],dsmodels['Parameter Grid'].iloc[i])


def calcAccuracyDS1(i):
    return classification(woNaNOutliers(ds), dsmodels['Model'].iloc[i], dsmodels['Parameter Grid'].iloc[i])


def calcAccuracyDS1(i):
    return classification(woOutliersMean(ds), dsmodels['Model'].iloc[i], dsmodels['Parameter Grid'].iloc[i])




