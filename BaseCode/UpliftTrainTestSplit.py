'''
Documentation:
    1. Used "uplift" as label(target)
    2. Dropped rows with "Non Value"
    3. Outlier searched (kept it simple with boxplot)
    4. Outlier killed with optimized IQR method
    5. After removing "Non Value" and "Outliers" there are still over 10.000 entries,
       that's why I decided to delete and not to scale
    6. Used train_test_split
    7. Printed the shape of train and test data
    8. Fitted the all models and accuracy
'''

import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

ds = get = pd.read_excel(r"C:\Users\Test\Desktop\SM\PromoEx_HTW_anonymized_data.xlsx")

db_pr_rf['mechanism_detailed'] = db_pr_rf.apply(
    lambda x: x.mechanism.replace('M', str(int(x.M))).replace('N', str(int(x.N))) \
        if x.mechanism in ['Dto NxM', 'Dto N+M'] \
        else x.mechanism, axis=1)

db_pr_rf['Discount'] = 1 - db_pr_rf['PN_old'] / db_pr_rf['PN_new']
db_pr_rf['month'] = db_pr_rf['start_date'].dt.month

# Print NaN sum
test = db_pr_rf.isnull().sum()

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
X_train, X_test, y_train, y_test = train_test_split(X_oneh, y, test_size=0.1, random_state=42)

print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Labels Shape:', y_test.shape)

# Random Forest
random_forest = RandomForestRegressor(n_estimators=100)
random_forest.fit(X_train, y_train)
predicted = random_forest.predict(X_test)
accuracy = random_forest.score(X_test, y_test)
print(accuracy)

# Decision Tree Algorithm
decision_model = DecisionTreeRegressor()
decision_model.fit(X_train, y_train)
predicted_decision_trees = decision_model.predict(X_test)
accuracy_decision_trees = decision_model.score(X_test, y_test)
print(accuracy_decision_trees)

# XGBoost algorithm
xg_model = GradientBoostingRegressor(n_estimators=100)
xg_model.fit(X_train, y_train)
predicted_XGBoost = xg_model.predict(X_test)
accuracy_XGBoost = xg_model.score(X_test, y_test)
print(accuracy_XGBoost)




