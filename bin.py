#outlier kill IQR method
iqr = 1.5 * (np.percentile(ds["ROI"], 60) - np.percentile(ds["ROI"], 40))
ds.drop(ds[ds["ROI"] > (iqr + np.percentile(ds["ROI"], 30))].index, inplace=True)
ds.drop(ds[ds["ROI"] < (np.percentile(ds["ROI"], 25) - iqr)].index, inplace=True)

print(ds2['ROI'].shape)




#uplift
import numpy as np

# remove nans and outliers
def woNaNOutliersU(ds):
    ds2 = ds.dropna(axis=0, how="any")

    out = []
    def Zscore_outlier(df):
        m = np.mean(df)
        sd = np.std(df)
        for i in df:
            z = (i - m) / sd
            if np.abs(z) > 3:
                out.append(i)
        print("Outliers:", out)

    Zscore_outlier(ds2["uplift"])

    for i in out:
        ds2.drop(index=next(iter(ds2[ds2['Uplift'] == i].index)), inplace=True)
    return ds2

# remove outliers and replace nans with mean
def woOutliersMeanU(ds):
    # outlier kill with Z-score
    out = []

    def Zscore_outlier(df):
        m = np.mean(df)
        sd = np.std(df)
        for i in df:
            z = (i - m) / sd
            if np.abs(z) > 3:
                out.append(i)
        print("Outliers:", out)

    Zscore_outlier(ds["uplift"])

    for i in out:
        ds.drop(index=next(iter(ds[ds['uplift'] == i].index)), inplace=True)

    ds4 = ds.fillna(ds.mean())

    return ds4


