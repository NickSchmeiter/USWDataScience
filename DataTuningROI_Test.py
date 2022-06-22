#TEST FILE FROM VIVI

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from DataSet import getDataset
from DataTuning import dropna, showBoxplot, zScore_outlier, boxPlotAfterOutliers

import scipy.stats as stats

#ds is full dataset with added columns
ds = getDataset()

#ds2 is dataset without NaN and outliers
ds2 = dropna(ds)
showBoxplot(ds2)

m = np.mean(ds)
sd = np.std(ds)
for i in ds:
    z = (i - m) / sd
    if np.abs(z) > 3:
        out.append(i)
print("Outliers:", out)

for i in out:
    ds2.drop(index=next(iter(ds[ds['ROI'] == i].index)), inplace=True)
    print(ds2.shape)
    plt.title("Box Plot after outlier removing")
    sns.boxplot(ds2["ROI"])
    plt.show()

'''
#ds2 is dataset without NaNs
ds2a=ds1.dropna(axis=0, how="any")
ds2=ds1.dropna(axis=0, how="any")

#ds3 is dataset without NaNs with replaced Mean
print('der Durchschnitt vom ROI beim ersten Datensatz ist:' + str(ds['ROI'].mean()))
ds3=ds1.fillna(ds1.mean())

#outlier search for ds2a
plt.figure
plt.title("Box Plot")
sns.boxplot(ds2a["ROI"])
plt.show()
print(ds2a.shape)
#outlier kill with Z-score
out=[]
def Zscore_outlier(df):
    m = np.mean(df)
    sd = np.std(df)
    for i in df:
        z = (i-m)/sd
        if np.abs(z) > 3:
            out.append(i)
    print("Outliers:",out)
Zscore_outlier(ds2a["ROI"])

for i in out:
    ds2a.drop(index = next(iter(ds2a[ds2a['ROI'] == i].index)), inplace=True)
print(ds2a.shape)
plt.title("Box Plot after outlier removing")
sns.boxplot(ds2a["ROI"])
plt.show()

#outlier search for ds3
plt.figure
plt.title("Box Plot")
sns.boxplot(ds3["ROI"])
plt.show()
print(ds3.shape)
#outlier kill with Z-score
out=[]
def Zscore_outlier(df):
    m = np.mean(df)
    sd = np.std(df)
    for i in df:
        z = (i-m)/sd
        if np.abs(z) > 3:
            out.append(i)
    print("Outliers:",out)
Zscore_outlier(ds3["ROI"])

for i in out:
    ds3.drop(index = next(iter(ds3[ds3['ROI'] == i].index)), inplace=True)
print(ds3.shape)
plt.title("Box Plot after outlier removing")
sns.boxplot(ds3["ROI"])
plt.show()

#outlier search for ds
ds4=ds
plt.figure
plt.title("Box Plot")
sns.boxplot(ds4["ROI"])
plt.show()
print(ds4.shape)
#outlier kill with Z-score
out=[]
def Zscore_outlier(df):
    m = np.mean(df)
    sd = np.std(df)
    for i in df:
        z = (i-m)/sd
        if np.abs(z) > 3:
            out.append(i)
    print("Outliers:",out)
Zscore_outlier(ds["ROI"])

for i in out:
    ds4.drop(index = next(iter(ds4[ds4['ROI'] == i].index)), inplace=True)
print(ds4.shape)
plt.title("Box Plot after outlier removing")
sns.boxplot(ds4["ROI"])
plt.show()
print('der Durchschnitt vom ROI beim ersten Datensatz4 ohne Outliers ist:' + str(ds4['ROI'].mean()))

#kill nan und fülle mit mean auf ds ohne outliers
ds4a=ds4.fillna(ds4.mean())
'''