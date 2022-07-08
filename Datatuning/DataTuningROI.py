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
    ds2 = ds.dropna(axis=0, how="any")

    out = []
    def Zscore_outlier(df):
        m = np.mean(df)
        sd = np.std(df)
        for i in df:
            z = (i - m) / sd
            if np.abs(z) > 3:
                out.append(i)

    Zscore_outlier(ds2["ROI"])

    for i in out:
        ds2.drop(index=next(iter(ds2[ds2['ROI'] == i].index)), inplace=True)

    return ds2

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
