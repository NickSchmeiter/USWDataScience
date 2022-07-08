import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV

from Datatuning.DataTuningROI import woNaN


def classification(ds,model,grid_param):
    ds['class'] = ds['class'].apply(lambda a: str(a).replace('low_impact', '2'))
    ds['class'] = ds['class'].apply(lambda a: str(a).replace('no_go', '1'))
    ds['class'] = ds['class'].apply(lambda a: str(a).replace('top_performer','5'))
    ds['class'] = ds['class'].apply(lambda a: str(a).replace('value_generator', '4'))
    ds['class'] = ds['class'].apply(lambda a: str(a).replace('volume_generator', '3'))
    ds=woNaN(ds)
    X = ds[['customer_lv_1', 'region_desc', 'canal_group', 'sku', 'mechanism_detailed', 'month', 'duration_consumer',
               'Discount', 'discount_so']]  # Features

    y = ds['class']  # Labels
    # One hot encode
    X_oneh = pd.get_dummies(X)

    # train-test-split
    X_train, X_test, y_train, y_test = train_test_split(X_oneh, y, test_size=0.8, random_state=42)

    # Grid Search!!!
    gd_sr = GridSearchCV(estimator=model,
                         param_grid=grid_param,
                         cv=5,
                         n_jobs=-1)
    gd_sr.fit(X_train, y_train)

    # accuracy_score
    return gd_sr.score(X_test, y_test)