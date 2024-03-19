from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import make_scorer
from ts_forecast.metric import InvestmentPortfolio

metric_business = InvestmentPortfolio()

def get_profit(y_true, y_pred):
  global metric_business
  return metric_business.get_score(y_true, y_pred)

get_score = make_scorer(get_profit)

def calibration(X, y):

      param_grid = {
          'n_estimators': [50, 100, 200],
          'max_depth': [3, 4, 5],
          'learning_rate': [0.01, 0.1, 0.5],
          'subsample': [0.5, 0.8, 1],
          'colsample_bytree': [0.5, 0.8, 1]
      }

      xgb = XGBRegressor()
      grid_search = GridSearchCV(xgb, param_grid=param_grid, cv=TimeSeriesSplit(n_splits=5), scoring=get_score)
      grid_search.fit(X, y)
      return grid_search
      