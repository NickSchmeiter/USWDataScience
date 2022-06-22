def woNaNOutliersU(ds):
    ds2 = ds.dropna(axis=0, how="any")

    plt.figure
    plt.title("Box Plot")
    sns.boxplot(ds2["uplift"])
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

    Zscore_outlier(ds2["uplift"])

    for i in out:
        ds2.drop(index=next(iter(ds2[ds2['Uplift'] == i].index)), inplace=True)
    print(ds2.shape)
    plt.title("Box Plot after outlier removing")
    sns.boxplot(ds2["uplift"])
    plt.show()
    return ds2


def woOutliersMeanU(ds):

    plt.figure
    plt.title("Box Plot")
    sns.boxplot(ds["uplift"])
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

    Zscore_outlier(ds["uplift"])

    for i in out:
        ds.drop(index=next(iter(ds[ds['uplift'] == i].index)), inplace=True)
    print(ds.shape)
    plt.title("Box Plot after outlier removing")
    sns.boxplot(ds["uplift"])
    plt.show()

    ds4 = ds.fillna(ds.mean())

    return ds4
