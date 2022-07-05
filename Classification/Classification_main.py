from sklearn.linear_model import LinearRegression

from Datatuning.DataSet import getDataset
from Classification.Classification_Pipe_Linear import classification
from Datatuning.DataTuningROI import woNaNOutliers, woOutliersMean,woNaN
from ModelDetails import getmodelcdf

dsmodels= getmodelcdf()

ds = getDataset()
i=1
def calcAccuracycDS1(i):
    return round(classification(woNaN(ds),dsmodels['Model'].iloc[i],dsmodels['Parameter Grid'].iloc[i]),4)*100


def calcAccuracycDS2(i):
    return round(classification(woNaNOutliers(ds), dsmodels['Model'].iloc[i], dsmodels['Parameter Grid'].iloc[i]),4)*100


def calcAccuracycDS3(i):
    return round(classification(woOutliersMean(ds), dsmodels['Model'].iloc[i], dsmodels['Parameter Grid'].iloc[i]),4)*100




