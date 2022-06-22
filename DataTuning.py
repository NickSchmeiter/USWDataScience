import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from DataSet import getDataset


def dropna (ds):
    ds.dropna(axis=0, how="any")


def replaceWithMean(ds):
    fillna(ds.mean())


def showBoxplot(ds):
    plt.figure
    plt.title("Box Plot")
    sns.boxplot(ds["ROI"])
    plt.show()
    print(ds.shape)


def zScore_outlier(ds):
    m = np.mean(ds)
    sd = np.std(ds)
    for i in ds:
        z = (i-m)/sd
        if np.abs(z) > 3:
            out.append(i)
    print("Outliers:", out)


def boxPlotAfterOutliers(ds):
    for i in out:
        ds.drop(index = next(iter(ds[ds['ROI'] == i].index)), inplace=True)
        print(ds.shape)
        plt.title("Box Plot after outlier removing")
        sns.boxplot(ds["ROI"])
        plt.show()





