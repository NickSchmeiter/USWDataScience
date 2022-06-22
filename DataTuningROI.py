import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from DataSet import getDataset


import scipy.stats as stats

ds = getDataset()

def woNaNOutliers(ds):
    ds2 = ds.dropna(axis=0, how="any")

    plt.figure
    plt.title("Box Plot")
    sns.boxplot(ds2["ROI"])
    plt.show()
    print(ds2.shape)

    out = []
    def Zscore_outlier(df):
        m = np.mean(df)
        sd = np.std(df)
        for i in df:
            z = (i - m) / sd
            if np.abs(z) > 3:
                out.append(i)
        print("Outliers:", out)

    Zscore_outlier(ds2["ROI"])

    for i in out:
        ds2.drop(index=next(iter(ds2[ds2['ROI'] == i].index)), inplace=True)
    print(ds2.shape)
    plt.title("Box Plot after outlier removing")
    sns.boxplot(ds2["ROI"])
    plt.show()
    return ds2


def woOutliersMean(ds):

    plt.figure
    plt.title("Box Plot")
    sns.boxplot(ds["ROI"])
    plt.show()
    print(ds.shape)

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

    Zscore_outlier(ds["ROI"])

    for i in out:
        ds.drop(index=next(iter(ds[ds['ROI'] == i].index)), inplace=True)
    print(ds.shape)
    plt.title("Box Plot after outlier removing")
    sns.boxplot(ds["ROI"])
    plt.show()

    ds4 = ds.fillna(ds.mean())

    return ds4
