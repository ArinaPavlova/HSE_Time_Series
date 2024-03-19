import numpy as np


class InvestmentPortfolio():
    """
    Class for simulation of business process 
    """
    def __init__(self, key_rate=0.07, day_deposit_rate=0.075, night_deposit_rate=0.061, night_loan_rate=0.08):
        self.key_rate = key_rate
        self.day_deposit_rate = day_deposit_rate
        self.night_deposit_rate = night_deposit_rate
        self.night_loan_rate = night_loan_rate

    def get_profit(self, trues, preds):
        trues = trues.copy()
        profit = np.zeros(trues.shape)

        # Начало дня - получаем прогноз модели
        positive_pred_mask = (preds > 0)
        # Принимаем решение вкладываться\нет
        profit[positive_pred_mask] += self.day_deposit_rate * preds[positive_pred_mask]
        # Изменение ликвидности после вложений
        trues[positive_pred_mask] -= preds[positive_pred_mask]
        # Покрываем дефицит ликвидности в течение дня, если прогноз < 0 из внутренних резервов
        trues[~positive_pred_mask] -= preds[~positive_pred_mask]

        # Конец дня - получение реального сальдо
        positive_balance_mask = (trues > 0)
        #Случай положительной ликвидности
        profit[positive_balance_mask] += self.night_deposit_rate * trues[positive_balance_mask]
        # Случай заема
        profit[~positive_balance_mask] += self.night_loan_rate * trues[~positive_balance_mask]

        return profit.sum()

    def get_score(self, trues, preds):
        profit = self.get_profit(trues, preds)
        return profit