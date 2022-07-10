import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

#regression method with parameters data, classifier and parameter grid
#return accuracy
def regression(ds, model, grid_param):
    #extract columns
    X = ds[['customer_lv_1', 'region_desc', 'canal_group', 'sku', 'mechanism_detailed', 'month', 'duration_consumer',
     'Discount', 'discount_so']]

    y = ds['ROI']  # Labels

    # One hot encode
    X_oneh = pd.get_dummies(X)

    # train-test-split
    X_train, X_test, y_train, y_test = train_test_split(X_oneh, y, test_size=0.2, random_state=42)

    # Grid Search!!!
    gd_sr = GridSearchCV(estimator=model,
                         param_grid=grid_param,
                         cv=5,
                         n_jobs=-1)
    gd_sr.fit(X_train, y_train)

    # accuracy_score
    accuracy = gd_sr.score(X_test, y_test)
    return accuracy
