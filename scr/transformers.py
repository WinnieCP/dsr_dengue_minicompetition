import pandas as pd
from sklearn.preprocessing import FunctionTransformer
import numpy as np

def select_features(df, features = None):
    # select features we want
    df_hlp = df[features].copy()
    return df_hlp

def deal_with_missing_values(df):
    # fill missing values with value from the past
    # we should probably make sure here that the dataframe
    # is ordered by time?
    df = df.fillna(method='ffill')
    return df

def select_city(df,city=None):
    # separate san juan and iquitos
    if 'city' in df.keys():
        df_city = df.loc[df.city == city]
        df_city = df_city.drop(columns = ['city'])
    else:
        df_city = df.loc[city]
    return df_city

def logtransform_target(target):
    return np.log(target+1.)
    
def inverse_logtransform_target(transformed):
    return np.exp(transformed)-1

def create_min_yearly_temp(x):
    if x.isnull().any().any()==True:
        x = deal_with_missing_values(x)
    x['min_yearly_temp']=x['station_min_temp_c'].rolling(53).min()
    x['min_yearly_temp']=x['min_yearly_temp'].fillna(x['min_yearly_temp'].mean())
    return x
    
def create_lagged_column(d,col_name,lag=1):
    df=d.copy()
    if df.isnull().any().any()==True:
        df = deal_with_missing_values(df)
    df[col_name+f'_lagged_{lag:1.2f}']=df[col_name].shift(-lag)
    df[col_name+f'_lagged_{lag:1.2f}'].fillna(method='ffill',inplace=True)
    return df


SelectCity = FunctionTransformer(func = select_city, check_inverse = False)
Impute = FunctionTransformer(func = deal_with_missing_values, check_inverse = False)
SelectFeatures = FunctionTransformer(func = select_features, check_inverse = False)