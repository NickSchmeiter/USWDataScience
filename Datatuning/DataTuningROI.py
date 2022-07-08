import numpy as np


out = []
def Zscore_outlier(df):
    m = np.mean(df)
    sd = np.std(df)
    for i in df:
        z = (i - m) / sd
        if np.abs(z) > 3:
            out.append(i)

# takes Dataset and drops NaN
# returns ds2
def woNaN(ds):
    ds2 = ds.dropna(axis=0, how="any")
    return ds2

# takes Dataset, drops NaN, finds outliers with Z-score and drops outliers
# returns ds2
def woNaNOutliers(ds):
    ds3 = ds.dropna(axis=0, how="any")

    out = []
    def Zscore_outlier(df):
        m = np.mean(df)
        sd = np.std(df)
        for i in df:
            z = (i - m) / sd
            if np.abs(z) > 3:
                out.append(i)

    Zscore_outlier(ds3["ROI"])

    for i in out:
        ds2.drop(index=next(iter(ds2[ds2['ROI'] == i].index)), inplace=True)

    return ds3

# takes Dataset, finds outliers with Z-score, drops outliers and replaces NaN with mean
# returns ds2
def woOutliersMean(ds):

    out = []
    def Zscore_outlier(df):
        m = np.mean(df)
        sd = np.std(df)
        for i in df:
            z = (i - m) / sd
            if np.abs(z) > 3:
                out.append(i)


    Zscore_outlier(ds["ROI"])

    for i in out:
        ds.drop(index=next(iter(ds[ds['ROI'] == i].index)), inplace=True)

    ds4 = ds.fillna(ds.mean())

    return ds4

# takes Dataset, drops NaNs, finds outliers with IQR, drops outliers and replaces NaN with mean
# returns ds2
# remove nans and outliers with IQR

def woOutliersIQR(ds):
    ds5 = ds.dropna(axis=0, how="any")

    iqr = 1.5 * (np.percentile(ds5["ROI"], 75) - np.percentile(ds5["ROI"], 25))
    ds5.drop(ds5[ds5["ROI"] > (iqr + np.percentile(ds5["ROI"], 30))].index, inplace=True)
    ds5.drop(ds5[ds5["ROI"] < (np.percentile(ds5["ROI"], 25) - iqr)].index, inplace=True)

    return ds5
