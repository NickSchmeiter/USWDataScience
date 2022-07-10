'''
Documentation:
    1. Used "ROI" as label(target)
    2. dropped rows with "Non Value"
    3. Outlier searched (kept it simple with boxplot)
    4. Outlier killed with optimized IQR method
    5. After removing "Non Value" and "Outliers" there are still over 10.000 entries,
       that's why I decided to delete and not to scale
    6. Used train_test_split
    7. Used for each model a GridSearchCV
    8. Printed for each model the best_estimators, best_cv_score, the final test score with new data
    9. Also printed the MSE and MAE to valuate the prediction
'''

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# import data
db_pr_rf = pd.read_excel(r"C:\Users\Test\Desktop\SM\PromoEx_HTW_anonymized_data.xlsx")

# create columns
db_pr_rf['mechanism_detailed'] = db_pr_rf.apply(
    lambda x: x.mechanism.replace('M', str(int(x.M))).replace('N', str(int(x.N))) \
        if x.mechanism in ['Dto NxM', 'Dto N+M'] \
        else x.mechanism, axis=1)

db_pr_rf['Discount'] = 1 - db_pr_rf['PN_old'] / db_pr_rf['PN_new']
db_pr_rf['month'] = db_pr_rf['start_date'].dt.month

# Print NaN sum
nan_sum = db_pr_rf.isnull().sum()
print(nan_sum)

# Drop na
db_pr = db_pr_rf.dropna(axis=0, how="any")

# Outlier searching
plt.figure(figsize=(10, 4))
plt.title("Box Plot")
sns.boxplot(db_pr["uplift"])
plt.show()

# Outlier killing Quelle => https://www.kaggle.com/code/nareshbhat/outlier-the-silent-killer/notebook

iqr = 1.5 * (np.percentile(db_pr["uplift"], 75) - np.percentile(db_pr["uplift"], 25))
db_pr.drop(db_pr[db_pr["uplift"] > (iqr + np.percentile(db_pr["uplift"], 30))].index, inplace=True)
db_pr.drop(db_pr[db_pr["uplift"] < (np.percentile(db_pr["uplift"], 25) - iqr)].index, inplace=True)
plt.title("Box Plot after outlier removing")
sns.boxplot(db_pr["uplift"])
plt.show()

# Info
print(db_pr["uplift"].describe())

X = db_pr[['customer_lv_1', 'region_desc', 'canal_group', 'sku', 'mechanism_detailed', 'month', 'duration_consumer',
           'Discount', 'discount_so']]  # Features
y = db_pr['uplift']  # Labels

# One hot encode
X_oneh = pd.get_dummies(X)

# train-test-split
X_train, X_test, y_train, y_test = train_test_split(X_oneh, y, test_size=0.2, random_state=42)

# using MSE(mean squared error) & MAE(mean_absolute_error )

# Grid Building Random Forest
grid_param_randomforest = {
    'min_samples_split': [2, 10, 100],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False],
    'n_estimators': [10, 100, 200],
    'max_depth': [10, 50, 100]

}
gd_sr_rf = GridSearchCV(estimator=RandomForestRegressor(),
                        param_grid=grid_param_randomforest,
                        cv=5, )
gd_sr_rf.fit(X_train, y_train)

print(" ")
print("RANDOMFORESTREGRESSOR:")
print('Best parameters: {}'.format(gd_sr_rf.best_params_))
print('Best cross-validation score: {:.2f}'.format(gd_sr_rf.best_score_))
print('Final Test Score with new data: {:.2f}'.format(gd_sr_rf.score(X_test, y_test)))
print("Test set MSE:", mean_squared_error(y_test, gd_sr_rf.predict(X_test)))
print("Test set MAE:", mean_absolute_error(y_test, gd_sr_rf.predict(X_test)))

# Grid Building DecisionTreeRegressor
grid_param_dt = {
    'min_samples_split': [2, 10, 20, 50],
    'splitter': ['best', 'random'],
    'max_features': ['auto', 'sqrt', 'log2']
}
gd_sr_dt = GridSearchCV(estimator=DecisionTreeRegressor(),
                        param_grid=grid_param_dt,
                        cv=5, )
gd_sr_dt.fit(X_train, y_train)

print(" ")
print("DECISIONTREEREGRESSOR:")
print('Best parameters: {}'.format(gd_sr_dt.best_params_))
print('Best cross-validation score: {:.2f}'.format(gd_sr_dt.best_score_))
print('Final Test Score with new data: {:.2f}'.format(gd_sr_dt.score(X_test, y_test)))
print("Test set MSE:", mean_squared_error(y_test, gd_sr_dt.predict(X_test)))
print("Test set MAE:", mean_absolute_error(y_test, gd_sr_dt.predict(X_test)))

# Grid Building LinearRegression
grid_param_lg = {
    'fit_intercept': [True, False],
    'copy_X': [True, False],
    'normalize': [True, False],
    'positive': [True, False]
}
gd_sr_lg = GridSearchCV(estimator=LinearRegression(),
                        param_grid=grid_param_lg,
                        cv=5, )
gd_sr_lg.fit(X_train, y_train)

print(" ")
print("LINEARREGRESSION: ")
print('Best parameters: {}'.format(gd_sr_lg.best_params_))
print('Best cross-validation score: {:.2f}'.format(gd_sr_lg.best_score_))
print('Final Test Score with new data: {:.2f}'.format(gd_sr_lg.score(X_test, y_test)))
print("Test set MSE:", mean_squared_error(y_test, gd_sr_lg.predict(X_test)))
print("Test set MAE:", mean_absolute_error(y_test, gd_sr_lg.predict(X_test)))

# Grild Building LinearSVR
grid_param_svr = {
    'C': [0.1, 1, 10],
    'max_iter': [1000, 10000, 100000]
}
gd_sr_svr = GridSearchCV(estimator=LinearSVR(),
                         param_grid=grid_param_svr,
                         cv=5, )
gd_sr_svr.fit(X_train, y_train)

print(" ")
print("LINEARSVR: ")
print('Best parameters: {}'.format(gd_sr_svr.best_params_))
print('Best cross-validation score: {:.2f}'.format(gd_sr_svr.best_score_))
print('Final Test Score with new data: {:.2f}'.format(gd_sr_svr.score(X_test, y_test)))
print("Test set MSE:", mean_squared_error(y_test, gd_sr_svr.predict(X_test)))
print("Test set MAE:", mean_absolute_error(y_test, gd_sr_svr.predict(X_test)))

# Grid Building GradientBoostingRegressor
grid_param_gbr = {
    'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
    'max_features': ['auto', 'sqrt', 'log2'],
    'n_estimators': [10, 100, 200],
    'min_samples_split': [2, 10, 20, 50]
}
gd_sr_gbr = GridSearchCV(estimator=GradientBoostingRegressor(),
                         param_grid=grid_param_gbr,
                         cv=5, )
gd_sr_gbr.fit(X_train, y_train)

print(" ")
print("GRADIENTBOOSTINGREGRESSOR:")
print('Best parameters: {}'.format(gd_sr_gbr.best_params_))
print('Best cross-validation score: {:.2f}'.format(gd_sr_gbr.best_score_))
print('Final Test Score with new data: {:.2f}'.format(gd_sr_gbr.score(X_test, y_test)))
print("Test set MSE:", mean_squared_error(y_test, gd_sr_gbr.predict(X_test)))
print("Test set MAE:", mean_absolute_error(y_test, gd_sr_gbr.predict(X_test)))
