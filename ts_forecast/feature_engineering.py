import pandas as pd
import numpy as np
from copy import deepcopy
import holidays
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import settings

def get_features(data: pd.DataFrame, 
                 holidays = None,
                 salary: bool = True, 
                 target: str = 'Balance', 
                 cols_extra: list = ['Nalog', 'Moex', 'Brent', 'Libor',
                                    'Rvi', 'Covid_cases', 'Covid_deaths', 
                                    'Key_rate', 'Inflation', 'Dollar',
                                    'Euro', ], 
                 additional_features = None, # features to add from tsfresh
                 date_based: bool = True, 
                 lags_day: bool = True, 
                 lags_month: bool = True, 
                 lags_week: bool = True,
                 lags_year: bool =  False,
                 window_weekdays: int = 7,
                 n_lags_day: int = 7,
                 n_lags_week: int = 4,
                 n_lags_month: int = 3,
                 n_lags_year: int = 1,
                 balance_aggregations_days: list = [3, 7, 14, 30],
                 factors_aggregations_days: list = []
                 ):
  
  """
  Function to extract time-based features from data
  """
  
  df = deepcopy(data[[target]+cols_extra])

  # rmv Nalog as it is binary
  if 'Nalog' in cols_extra:
    cols_extra.remove('Nalog')

  # -------------------
  # additional features

  # a montn frequency
  for col in cols_extra:
    if col in ('Inflation',):
      df[col] = df[col].shift(30)   
    else:
      # shift 1
      df[col] = df[col].shift(1)

  # it didnt work without a column

  # -------------------
  # special dates
  # salary days 
  df['salary_payments'] = df.index.day.isin([5, 20]).astype(int)
  # holidays
  if holidays:
    df['holidays'] = df.index.isin(holidays).astype(int)

  # -------------------
  # date based featuress
  if date_based:

    # binary / ordinal

    # weekends
    df['weekends'] = df.index.day_of_week.isin([5, 6]).astype(int)  
    # categorical 
    df['first_day_of_month'] = (df.index.day==1).astype(int)

    # categories

    # day of week
    weekday = pd.get_dummies(df.index.weekday, prefix='weekday').set_index(df.index)

    # month
    month = pd.get_dummies(df.index.month, prefix='month').set_index(df.index)

    # quarter
    quarter = pd.get_dummies(df.index.quarter, prefix='quarter').set_index(df.index)

    features_df = pd.concat([weekday, month, quarter],axis=1)
    df = pd.concat([df, features_df], axis=1)
    
    # agregations by week days (seasonality)
    if window_weekdays is not None:
            df['balance_rolling_mean_weekday'] = df.groupby(data.index.weekday)[target].transform(lambda x: x.rolling(window_weekdays).mean().shift(1))
            df['balance_rolling_max_weekday'] = df.groupby(data.index.weekday)[target].transform(lambda x: x.rolling(window_weekdays).max().shift(1))
            df['balance_rolling_min_weekday'] = df.groupby(data.index.weekday)[target].transform(lambda x: x.rolling(window_weekdays).min().shift(1))
            df['balance_rolling_median_weekday'] = df.groupby(data.index.weekday)[target].transform(lambda x: x.rolling(window_weekdays).median().shift(1))
            df['balance_rolling_std_weekday'] = df.groupby(data.index.weekday)[target].transform(lambda x: x.rolling(window_weekdays).std().shift(1))

  # -------------------
  # lags 
  # days
  if lags_day: 
    for i in range(n_lags_day):
      df[f'balance_lag_day_{i + 1}'] = df[target].shift(i + 1)
  
  # weeks
  if lags_week: 
    for i in range(1, n_lags_week):
      df[f'balance_lag_week_{i + 1}'] = df[target].shift((i + 1)*7)
  
  # months 
  if lags_month: 
    for i in range(n_lags_month):
      df[f'balance_lag_month_{i + 1}'] = df[target].shift((i + 1)*30)

  # years 
  if lags_year: 
    for i in range(n_lags_year):
      df[f'balance_lag_year_{i + 1}'] = df[target].shift((i + 1)*365)

  # -------------------
  # aggregations by certain periods
  for days in balance_aggregations_days:
    rolling_ = data[target].rolling(days).agg(['mean','std','median','max','min']).shift(1)
    rolling_.columns = ['_'.join(col) + f'_{days}' if type(col) == tuple else col + f'_{days}' for col in rolling_.columns]
    df = pd.concat([df, rolling_], axis=1)

  for days in factors_aggregations_days:
    rolling_ = data[cols_extra].rolling(days).agg(['mean','std','median','max','min']).shift(1)
    rolling_.columns = ['_'.join(col) + f'_{days}' if type(col) == tuple else col + f'_{days}' for col in rolling_.columns]
    df = pd.concat([df, rolling_], axis=1)

  # -------------------
  # features from tfresh
  # if some characteristics are useful, we may add them

  if additional_features:
      extraction_settings = settings.EfficientFCParameters()
      X = extract_features(data[[target]].reset_index().reset_index(), 
                          column_id='index',
                          column_value='Balance', column_sort='Date',
                          default_fc_parameters=extraction_settings,
                          # we impute = remove all NaN features automatically
                          impute_function=impute)

      X = X[additional_features]
      X.index = df.index
      df = pd.concat([df, X], axis=1)

  df = df[n_lags_month*30:]
  df = df[30:]

  return df


def impute_nans(df):
    df['Euro'] = df['Euro'].replace(0, np.nan)
    df.ffill(inplace=True)
    