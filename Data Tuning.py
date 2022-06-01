import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


#ds1 is full dataset with added columns
import scipy.stats as stats

ds = pd.read_excel(r"C:\Users\nicks\Downloads\PromoEx_HTW_anonymized_data.xlsx")
ds['mechanism_detailed'] = ds.apply(
    lambda x: x.mechanism.replace('M', str(int(x.M))).replace('N', str(int(x.N))) \
        if x.mechanism in ['Dto NxM', 'Dto N+M'] \
        else x.mechanism, axis=1)

#ds2 and ds3 drop duplicates
ds1=ds.drop_duplicates()

#ds2 is dataset without NaNs
ds2=ds1.dropna(axis=0, how="any")

#ds3 is dataset without NaNs with replaced Mean
ds3=ds1.fillna(ds1.mean())

print(ds.shape)
print(ds1.shape)
print(ds2.shape)
print(ds3.shape)


#outlier search
plt.figure
plt.title("Box Plot")
sns.boxplot(ds["ROI"])
plt.show()
print(ds['ROI'].shape)
#outlier kill with Z-score
out=[]
def Zscore_outlier(df):
    m = np.mean(df)
    sd = np.std(df)
    for i in df:
        z = (i-m)/sd
        if np.abs(z) > 1:
            out.append(i)
    print("Outliers:",out)
Zscore_outlier(ds2["ROI"])

for i in out:
    ds2.drop(index = next(iter(ds2[ds2['ROI'] == i].index)), inplace=True)
print(ds['ROI'].shape)
plt.title("Box Plot after outlier removing")
sns.boxplot(ds["ROI"])
plt.show()



