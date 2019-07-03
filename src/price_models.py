import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

def drop_unneeded_columns(df,col_list):
    df.drop(col_list,axis=1,inplace=True)
    return df

def get_dummies(df,cols):
    for i in cols:
        df = pd.get_dummies(df, columns=[i])
    return df

def origin_simplification(df,origins_to_fill):
    df['origin'].where(df['origin'].isin(origins_to_fill), 'Other', inplace=True)
    return df

def varietal_simplification(df,varietals_to_fill):
    df['varietal'].where(df['varietal'].isin(varietals_to_fill), 'Other', inplace=True)
    return df


def sm_linear_regression(X_train,y_train):
    model = sm.OLS(y_train,sm.add_constant(X_train))
    results = model.fit()
    
    print(results.rsquared_adj)
    print(results.summary())


def sklearn_linear_regression(X_train,y_train,X_test,y_test):
    model = LinearRegression()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    print('Linear Regression rsquared: {}'.format(r2_score(y_test,y_pred)))
    print('Linear Regression rmse: {}'.format(np.sqrt(mean_squared_error(y_test,y_pred))))

def random_forest(X_train,y_train,X_test,y_test):
    model = RandomForestRegressor(n_estimators=200,max_depth=10)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    print('Random Forest rsquared: {}'.format(model.score(X_test,y_test)))
    print('Random Forest rmse: {}'.format(np.sqrt(mean_squared_error(y_test,y_pred))))

if __name__ == '__main__':
    df = pd.read_csv('../data/50x50/sorted_df.csv')
    col_list = ['url','avg_rating',
        'page','rating_dict','price_bins','name',
        'pc7','pc6','pc5']
    df = drop_unneeded_columns(df,col_list)


    # origins_to_fill = ['France','Italy','California']
    # origin_simplification(df,origins_to_fill)
    varietals_to_fill = ['Pinot Noir','Chardonnay',
      'Cabernet Sauvignon','Bordeaux Red Blends',
      'Other Red Blends','Sauvignon Blanc']    
    # varietals_to_fill = ['Bordeaux Red Blends']
    varietal_simplification(df,varietals_to_fill)
    
    cols = ['origin','varietal']
    df = get_dummies(df,cols)

    y = np.log(df.pop('price'))
    X_train, X_test, y_train, y_test = train_test_split(df, y)

    sm_linear_regression(X_train,y_train)
    sklearn_linear_regression(X_train,y_train,X_test,y_test)
    random_forest(X_train,y_train,X_test,y_test)
    