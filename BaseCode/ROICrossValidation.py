'''
Documentation: (Steps)
    1. Used "uplift" as label(target)
    2. dropped rows with "Non Value"
    3. Outlier searched (kept it simple with boxplot)
    4. Outlier killed with optimized IQR method
    5. After removing "Non Value" and "Outliers" there are still over 10.000 entries,
       that's why I decided to delete and not to scale
    6. Used train_test_split
    7. Prepared a pipeline for cross validation
    8 Used the cross-validation on all created pipelines with different regressors
    9. Fitted the models with the training data
    10. Printed the sklearn metrics r2_score and MSE for all models to get the accuracy and to check if the model is overfitted
'''

import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.svm import SVR

db_pr_rf = pd.read_excel(r"C:\Users\Test\Desktop\SM\PromoEx_HTW_anonymized_data.xlsx")

db_pr_rf['mechanism_detailed'] = db_pr_rf.apply(
    lambda x: x.mechanism.replace('M', str(int(x.M))).replace('N', str(int(x.N))) \
        if x.mechanism in ['Dto NxM', 'Dto N+M'] \
        else x.mechanism, axis=1)

db_pr_rf['Discount'] = 1 - db_pr_rf['PN_old'] / db_pr_rf['PN_new']
db_pr_rf['month'] = db_pr_rf['start_date'].dt.month

# Print NaN sum
test = db_pr_rf.isnull().sum()
print(test)
# Drop na
db_pr = db_pr_rf.dropna(axis=0, how="any")

# Outlier searching
plt.figure(figsize=(10, 4))
plt.title("Box Plot")
sns.boxplot(db_pr["ROI"])
plt.show()

# Outlier killing Quelle => https://www.kaggle.com/code/nareshbhat/outlier-the-silent-killer/notebook

iqr = 1.5 * (np.percentile(db_pr["ROI"], 75) - np.percentile(db_pr["ROI"], 25))
db_pr.drop(db_pr[db_pr["ROI"] > (iqr + np.percentile(db_pr["ROI"], 30))].index, inplace=True)
db_pr.drop(db_pr[db_pr["ROI"] < (np.percentile(db_pr["ROI"], 25) - iqr)].index, inplace=True)
plt.title("Box Plot after outlier removing")
sns.boxplot(db_pr["ROI"])
plt.show()

# Info
print(db_pr["ROI"].describe())

X = db_pr[['customer_lv_1', 'region_desc', 'canal_group', 'sku', 'mechanism_detailed', 'month', 'duration_consumer',
           'Discount', 'discount_so']]  # Features
y = db_pr['ROI']  # Labels

# One hot encode
X_oneh = pd.get_dummies(X)

# train-test-split
X_train, X_test, y_train, y_test = train_test_split(X_oneh, y, test_size=0.1, random_state=42)

print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Labels Shape:', y_test.shape)

# pipeline
my_pipeline1 = make_pipeline(SimpleImputer(), RandomForestRegressor(n_estimators=100))
my_pipeline2 = make_pipeline(SimpleImputer(), GradientBoostingRegressor())
my_pipeline3 = make_pipeline(SimpleImputer(), DecisionTreeRegressor())
my_pipeline4 = make_pipeline(SimpleImputer(), SVR())

# https://www.kaggle.com/code/dansbecker/cross-validation/notebook

k_fold_acc1 = cross_val_score(my_pipeline1, X_train, y_train, cv=5)
print(k_fold_acc1)

k_fold_acc2 = cross_val_score(my_pipeline2, X_train, y_train, cv=5)
print(k_fold_acc2)

k_fold_acc3 = cross_val_score(my_pipeline3, X_train, y_train, cv=5)
print(k_fold_acc3)

k_fold_acc4 = cross_val_score(my_pipeline4, X_train, y_train, cv=5)
print(k_fold_acc4)

# randomforest r2_score and MSE
final_rf = my_pipeline1.fit(X_train, y_train)
print("RandomForest: ")
print("Test set MSE:", mean_squared_error(y_test, final_rf.predict(X_test)))
print("Test set r2_score:", r2_score(y_test, final_rf.predict(X_test)))
print(" ")

# GradientBoosting r2_score and MSE
final_gd = my_pipeline2.fit(X_train, y_train)
print("GradoemtBoosting: ")
print("Test set MSE:", mean_squared_error(y_test, final_gd.predict(X_test)))
print("Test set r2_score:", r2_score(y_test, final_gd.predict(X_test)))

# DecisionTree r2_score and MSE
final_dt = my_pipeline3.fit(X_train, y_train)
print("DecisionTree: ")
print("Test set MSE:", mean_squared_error(y_test, final_dt.predict(X_test)))
print("Test set r2_score:", r2_score(y_test, final_dt.predict(X_test)))

# SVR r2_score and MSE
final_svr = my_pipeline4.fit(X_train, y_train)
print("SVR: ")
print("Test set MSE:", mean_squared_error(y_test, final_svr.predict(X_test)))
print("Test set r2_score:", r2_score(y_test, final_svr.predict(X_test)))





