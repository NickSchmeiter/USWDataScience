from ModelDetails import getmodelrdf
from DataSet import getDataset
from RegressionPipeLinear import regression
from DataTuningROI import woNaN, woNaNOutliers, woOutliersMean

dsmodels= getmodelrdf()
ds = getDataset()


def calcAccuracyDS1(i):
    return regression(woNaN(ds), dsmodels['Model'].iloc[i], dsmodels['Parameter Grid'].iloc[i])


def calcAccuracyDS2(i):
    return regression(woNaNOutliers(ds), dsmodels['Model'].iloc[i], dsmodels['Parameter Grid'].iloc[i])


def calcAccuracyDS3(i):
    return regression(woOutliersMean(ds), dsmodels['Model'].iloc[i], dsmodels['Parameter Grid'].iloc[i])

