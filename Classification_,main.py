from sklearn.linear_model import LinearRegression

from DataSet import getDataset
from Classification_Pipe_Linear import classification
from DataTuningROI import woNaNOutliers, woOutliersMean,woNaN
from ModelDetails import getmodeldf

dsmodels= getmodeldf()

ds = getDataset()

classification(woNaN(ds),dsmodels['Model'].iloc[2]
,dsmodels['Parameter Grid'].iloc[2])
classification(woNaNOutliers(ds),dsmodels['Model'].iloc[2]
,dsmodels['Parameter Grid'].iloc[2])
classification(woOutliersMean(ds),dsmodels['Model'].iloc[2]
,dsmodels['Parameter Grid'].iloc[2])

