import matplotlib.pyplot as plt
from Datatuning.DataSet import getDataset
import seaborn as sns
import numpy as np

# boxplot for outlier search
# file is not used anymore
ds = getDataset()

plt.figure
plt.title("Box Plot")
sns.boxplot(ds["ROI"])
plt.show()
print(ds['ROI'].shape)

# outlier kill with Z-score
out = []


def Zscore_outlier(df):
    m = np.mean(df)
    sd = np.std(df)
    for i in df:
        z = (i - m) / sd
        if np.abs(z) > 1:
            out.append(i)
    print("Outliers:", out)


Zscore_outlier(ds["ROI"])

for i in out:
    ds.drop(index=next(iter(ds[ds['ROI'] == i].index)), inplace=True)
print(ds['ROI'].shape)
plt.title("Box Plot after outlier removing")
sns.boxplot(ds["ROI"])
plt.show()
