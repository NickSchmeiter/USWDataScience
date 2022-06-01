#outlier kill IQR method
iqr = 1.5 * (np.percentile(ds["ROI"], 60) - np.percentile(ds["ROI"], 40))
ds.drop(ds[ds["ROI"] > (iqr + np.percentile(ds["ROI"], 30))].index, inplace=True)
ds.drop(ds[ds["ROI"] < (np.percentile(ds["ROI"], 25) - iqr)].index, inplace=True)

print(ds2['ROI'].shape)



