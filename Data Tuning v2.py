import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


#ds1 is full dataset with added columns
ds1 = pd.read_excel(r"C:\Users\nicks\Downloads\PromoEx_HTW_anonymized_data.xlsx")
ds1['mechanism_detailed'] = ds1.apply(
    lambda x: x.mechanism.replace('M', str(int(x.M))).replace('N', str(int(x.N))) \
        if x.mechanism in ['Dto NxM', 'Dto N+M'] \
        else x.mechanism, axis=1)
#ds2 is dataset without NaNs
ds2=ds1.dropna(axis=0, how="any")
#ds3 is dataset without NaNs with replaced Mean
ds3=ds1.fillna(ds1.mean())
#ds2 and ds3 drop duplicates
ds2=ds2.drop_duplicates()
ds3=ds3.drop_duplicates()
#outlier search
plt.figure(figsize=(10, 4))
plt.title("Box Plot")
sns.boxplot(ds1["uplift"])
plt.show()
#outlier kill
iqr = 1.5 * (np.percentile(ds1["uplift"], 60) - np.percentile(ds1["uplift"], 40))
ds1.drop(ds1[ds1["uplift"] > (iqr + np.percentile(ds1["uplift"], 30))].index, inplace=True)
ds1.drop(ds1[ds1["uplift"] < (np.percentile(ds1["uplift"], 25) - iqr)].index, inplace=True)
plt.title("Box Plot after uplift outlier removing")
sns.boxplot(ds1["uplift"])
plt.show()