import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

#ds is full dataset with added columns
ds = pd.read_excel(r"C:\Users\nicks\Downloads\PromoEx_HTW_anonymized_data.xlsx")
ds['mechanism_detailed'] = ds.apply(
   lambda x: x.mechanism.replace('M', str(int(x.M))).replace('N', str(int(x.N))) \
        if x.mechanism in ['Dto NxM', 'Dto N+M'] \
        else x.mechanism, axis=1)

print('Der Durchschnitt vom Uplift bei ds: ' + str(ds['uplift'].mean()))

#ds2 is ds without duplicates
ds1=ds.drop_duplicates()
print("Dataset without duplicates " + str(ds1.shape))


#ds2 is dataset without NaNs
ds2=ds.dropna(axis=0, how="any")
print("Dataset without NaN " + str(ds2.shape))

#ds2 outlier kill
#returns dataset without NaN and outliers
plt.figure
plt.title("Box Plot uplift with Outliers for ds2")
sns.boxplot(ds2["uplift"])
plt.show()

out=[]
def Zscore_outlier(df):
    m = np.mean(df)
    sd = np.std(df)
    for i in df:
        z = (i-m)/sd
        if np.abs(z) > 3:
            out.append(i)
    print("Outliers:",out)
Zscore_outlier(ds2["uplift"])

for i in out:
    ds2.drop(index = next(iter(ds2[ds2['uplift'] == i].index)), inplace=True)
print("Dataset ds2 without NaN and Outliers " + str(ds2.shape))
plt.title("Box Plot after outlier removing for d2")
sns.boxplot(ds2["uplift"])
plt.show()


#ds3 is dataset where NaNs were replaced by mean
ds3=ds.fillna(ds.mean())
print("Dataset where NaNs were replaced by mean " + str(ds3.shape))

#ds3 outlier kill
#returns dataset with NaN replaced by mean and without outliers
plt.figure
plt.title("Box Plot uplift with Outliers for ds2")
sns.boxplot(ds3["uplift"])
plt.show()

out=[]
def Zscore_outlier(df):
    m = np.mean(df)
    sd = np.std(df)
    for i in df:
        z = (i-m)/sd
        if np.abs(z) > 3:
            out.append(i)
    print("Outliers:",out)
Zscore_outlier(ds3["uplift"])

for i in out:
    ds3.drop(index = next(iter(ds3[ds3['uplift'] == i].index)), inplace=True)
print("Dataset ds3 without replaced NaN and without Outliers " + str(ds3.shape))
plt.title("Box Plot after outlier removing for d3")
sns.boxplot(ds3["uplift"])
plt.show()



#outlier kill for ds, saved in ds4
#returns dataset ds4 without outliers

ds4 = ds

plt.figure
plt.title("Box Plot uplift with Outliers for ds4")
sns.boxplot(ds4["uplift"])
plt.show()

out=[]
def Zscore_outlier(df):
    m = np.mean(df)
    sd = np.std(df)
    for i in df:
        z = (i-m)/sd
        if np.abs(z) > 3:
            out.append(i)
    print("Outliers:",out)
Zscore_outlier(ds4["uplift"])

for i in out:
    ds4.drop(index = next(iter(ds4[ds4['uplift'] == i].index)), inplace=True)
print("Dataset ds4 - without outliers " + str(ds4.shape))
plt.title("Box Plot after outlier removing for d4")
sns.boxplot(ds4["uplift"])
plt.show()

print('Der Durchschnitt vom Uplift bei ds4: ' + str(ds4['uplift'].mean()))

#ds4 ohne Outliers, NaNs ersetzt durch mean
ds4a=ds4.fillna(ds4.mean())


