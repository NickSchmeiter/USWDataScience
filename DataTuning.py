from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

def dropna (ds):
    ds.dropna(axis=0, how="any")


def replaceWithMean(ds):
    ds.fillna(ds.mean())


def showBoxplot(ds):
    plt.figure
    plt.title("Box Plot")
    sns.boxplot(ds["ROI"])
    plt.show()
    print(ds.shape)





