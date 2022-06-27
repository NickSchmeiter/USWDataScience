from DataSet import getDataset
from Classification_Pipe_Linear import classification
from DataTuningROI import woNaNOutliers, woOutliersMean

ds = getDataset()

classification(ds)
classification(woNaNOutliers(ds))
classification(woOutliersMean(ds))

