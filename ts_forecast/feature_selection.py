import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from fcbf import fcbf
from rapidfuzz.distance import Indel
from rapidfuzz.distance import Hamming
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
import statistics
from ts_forecast.metric import InvestmentPortfolio
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class Feature_Selection():
    """
    Class for relevant features according to different methods
    """

    def __init__(self, min_features_to_select):
        self.min_features_to_select = min_features_to_select    

    def transform(self, X, y):
        res1 = self.redulariz_L1(X, y)   
        res2 = self.fcbf(X, y)
        res3 = self.rfecv_check(X, y)
        return res1, res2,  res3

    def redulariz_L1(self, X, y):
        lasso = Lasso()
        param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
        tscv = TimeSeriesSplit(n_splits = 3)
        grid_search = GridSearchCV(lasso, param_grid, cv=tscv.split(X)) 
        

        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        lasso = Lasso(alpha=best_params['alpha'])
        lasso.fit(X, y)
        lasso_coef = pd.DataFrame({'Feature': X.columns, 'Coefficient': lasso.coef_})
        lasso_coef = lasso_coef.sort_values(by='Coefficient', ascending=False)

        conditions = [lasso_coef['Coefficient'] != 0]
        selected_rows = lasso_coef[conditions[0]]
        selected_features = selected_rows['Feature'].tolist()
        return list(selected_features)

    def fcbf(self, X, y):
        relevant_features, irrelevant_features, correlations = fcbf(X, y, su_threshold=0.2, base=2)
        return list(relevant_features)

    
    def rfecv_check(self, X, y):
        clf = LinearRegression()
        rfecv = RFECV(
            estimator=clf,
            step=1,
            min_features_to_select=self.min_features_to_select,
            n_jobs=2,
        )
        rfecv.fit(X, y)
        return list(X.columns[rfecv.support_])


class Model_Selection():

    """
    Class to tune models to find best combo of params
    """
    def __init__(self):
        pass    

    def transform(self, X_train, y_train, X_test, y_test):
        res1 = self.log_reg(X_train, y_train, X_test, y_test)   
        res2 = self.svm(X_train, y_train, X_test, y_test)
        return res1, res2

    def log_reg(self, X_train, y_train, X_test, y_test):
        model = LinearRegression(fit_intercept=True)
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        return (prediction, mean_absolute_error(y_test, prediction))

    def svm(self, X_train, y_train, X_test, y_test):
        svm = SVR()
        param_grid = {'C': [0.1, 1, 10, 100, 1000], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']} 
        grid = GridSearchCV(svm, param_grid, cv=TimeSeriesSplit(n_splits=5), error_score='raise')
        grid.fit(X_train, y_train)
        prediction = grid.predict(X_test)
        return (prediction, mean_absolute_error(y_test, prediction))

    def xg_boost(self, X_train, y_train, X_test, y_test):
      param_grid = {
          'n_estimators': [50, 100, 200],
          'max_depth': [3, 4, 5],
          'learning_rate': [0.01, 0.1, 0.5],
          'subsample': [0.5, 0.8, 1],
          'colsample_bytree': [0.5, 0.8, 1]
      }

      xgb = XGBRegressor()
      grid_search = GridSearchCV(xgb, param_grid=param_grid, cv=TimeSeriesSplit(n_splits=5), scoring='neg_mean_absolute_error')
      grid_search.fit(X_train, y_train)
      y_pred = grid_search.predict(X_test)
      return (y_pred, mean_absolute_error(y_test, y_pred))


def stable_features(X, y):
    """
    Function to calculate model stability, quality and business profit
    for choosing the best option
    """

    def avg_index_calculation(stab_features):
        pairwise_index = []
        for i in range(len(stab_features)):
            for j in range(i+1, len(stab_features)):     
                index_i = Hamming.normalized_similarity(stab_features[i], stab_features[j])
                # index_i = Indel.normalized_similarity(stab_features[i], stab_features[j])
                pairwise_index.append(index_i)
        return statistics.mean(pairwise_index)

    def model_results_f1(X_train, y_train, X_test, y_test, l1_list, df_stat):
        fs = Model_Selection()
        inv_p = InvestmentPortfolio()
        for i in range(0, len(l1_list)):
          columns = l1_list[i]
          if len(columns) > 0:
              X_te = X_test.loc[:, columns]
              res1 = fs.log_reg(X_train.loc[:, columns], y_train, X_te.loc[:, columns], y_test)
              df_stat.loc[len(df_stat)] = [l1_list[i], 'LogReg',  avg_index_calculation(l1_list), res1[1], inv_p.get_score(y_test, res1[0])]
              res2 = fs.svm(X_train.loc[:, columns], y_train, X_te.loc[:, columns], y_test)
              df_stat.loc[len(df_stat)] = [l1_list[i], 'SVM',  avg_index_calculation(l1_list), res2[1], inv_p.get_score(y_test, res2[0])]
              res3 = fs.xg_boost(X_train.loc[:, columns], y_train, X_te.loc[:, columns], y_test)
              df_stat.loc[len(df_stat)] = [l1_list[i], 'Xgboost',  avg_index_calculation(l1_list), res3[1], inv_p.get_score(y_test, res3[0])]
        return df_stat


    l1_list, fcbf_list, rfe_list  = [], [], []
    fs = Feature_Selection(min_features_to_select=6)
    stab_features = []
    tscv = TimeSeriesSplit(gap=0, max_train_size=None, n_splits=5, test_size=None)
    for train_index, test_index in tscv.split(X):
        X_val = X.iloc[np.array(train_index[-257:])]
        y_val = y.iloc[np.array(train_index[-257:])]
        L1, fcbf_res, rfe  =  fs.transform(X_val, y_val)

        l1_list.append(L1)
        fcbf_list.append(fcbf_res)
        rfe_list.append(rfe)
    # print(l1_list)
    feat_stab_result = dict()
    feat_stab_result['L1'] = avg_index_calculation(l1_list)
    feat_stab_result['FCBF'] = avg_index_calculation(fcbf_list)
    feat_stab_result['RFECV'] = avg_index_calculation(rfe_list)
    df_stat = pd.DataFrame(columns=['Features', 'Model', 'Stability', 'MAE', 'Business'])
    test_size=0.3
    test_index = int(len(X)*(1 - test_size))
    X_train = X.iloc[:test_index]
    y_train = y.iloc[:test_index]
    X_test = X.iloc[test_index:]
    y_test = y.iloc[test_index:]
    df_stat = model_results_f1(X_train, y_train, X_test, y_test, l1_list, df_stat)
    df_stat = model_results_f1(X_train, y_train, X_test, y_test, fcbf_list, df_stat)
    df_stat = model_results_f1(X_train, y_train, X_test, y_test, rfe_list, df_stat)
    df_stat['Business_norm'] = (df_stat['Business'] - df_stat['Business'].min()) / (df_stat['Business'].max() - df_stat['Business'].min())
    df_stat['MAE_norm'] = (df_stat['MAE'] - df_stat['MAE'].min()) / (df_stat['MAE'].max() - df_stat['MAE'].min())
    df_stat['Metrics_range'] = (0.5 * (1 - df_stat['MAE_norm']) + 0.25 * df_stat['Stability'] + 0.25 *df_stat['Business_norm']) / 3
    res_model = df_stat[df_stat['Metrics_range'] == df_stat['Metrics_range'].max()].reset_index()
    return res_model['Features'][0], res_model['Model'][0], df_stat
