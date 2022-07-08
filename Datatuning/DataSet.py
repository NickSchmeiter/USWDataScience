import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def getDataset():
    #ds = pd.read_excel(r"C:\Users\nicks\Downloads\PromoEx_HTW_anonymized_data.xlsx")
    #ds = pd.read_excel(r"C:\Users\Schmeiter\Downloads\PromoEx_HTW_anonymized_data.xlsx")
    ds = pd.read_excel("/Users/viviennelamboy/Downloads/PromoEx_HTW_anonymized_data.xlsx")

    ds['mechanism_detailed'] = ds.apply(
        lambda x: x.mechanism.replace('M', str(int(x.M))).replace('N', str(int(x.N))) \
            if x.mechanism in ['Dto NxM', 'Dto N+M'] \
            else x.mechanism, axis=1)
    ds.drop_duplicates()
    ds['Discount'] = 1 - ds['PN_old'] / ds['PN_new']
    ds['month'] = ds['start_date'].dt.month

    ds = ds[0:10]

    return ds