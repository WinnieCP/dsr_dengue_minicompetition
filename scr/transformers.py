import pandas as pd
from sklearn.preprocessing import FunctionTransformer


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

SelectCity = FunctionTransformer(func = select_city, check_inverse = False)
Impute = FunctionTransformer(func = deal_with_missing_values, check_inverse = False)
SelectFeatures = FunctionTransformer(func = select_features, check_inverse = False)