from sklearn.linear_model import LinearRegression

from Datatuning.DataSet import getDataset
from Classification.Classification_Pipe_Linear import classification
from Datatuning.DataTuningROI import woNaNOutliers, woOutliersMean,woNaN,woOutliersIQR
from ModelDetails import getmodelcdf
#gets df with models and grid params
dsmodels= getmodelcdf()
#gets dataset
ds = getDataset()
#calls classification methods for first ds and rounds
def calcAccuracycDS1(i):
    return round(classification(woNaN(ds),dsmodels['Model'].iloc[i],dsmodels['Parameter Grid'].iloc[i]),4)*100

#calls classification methods for second ds and rounds
def calcAccuracycDS2(i):
    return round(classification(woNaNOutliers(ds), dsmodels['Model'].iloc[i], dsmodels['Parameter Grid'].iloc[i]),4)*100

#calls classification methods for third ds and rounds
def calcAccuracycDS3(i):
    return round(classification(woOutliersMean(ds), dsmodels['Model'].iloc[i], dsmodels['Parameter Grid'].iloc[i]),4)*100

#calls classification methods for fourth ds and rounds
def calcAccuracycDS4(i):
    return round(classification(woOutliersIQR(ds),dsmodels['Model'].iloc[i],dsmodels['Parameter Grid'].iloc[i]),4)*100

