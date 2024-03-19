import numpy as np
from scipy.stats import norm
from statistics import median


#Класс, реализующий статистику для обнаружения разладки
class Stat(object):
    def __init__(self, threshold, direction="unknown", init_stat=0.0):
        self._direction = str(direction)
        self._threshold = float(threshold)
        self._stat = float(init_stat)
        self._alarm = self._stat / self._threshold
    
    @property
    def direction(self):
        return self._direction

    @property
    def stat(self):
        return self._stat
        
    @property
    def alarm(self):
        return self._alarm
        
    @property
    def threshold(self):
        return self._threshold
    
    def update(self, **kwargs):
        self._alarm = self._stat / self._threshold


def normal_likelihood(value, mean_0, mean_8, std):
    return np.log(norm.pdf(value, mean_0, std) / 
                  norm.pdf(value, mean_8, std))


#Статистика кумулятивных сумм для обнаружения разладки среднего значения
class AdjustedCusum(Stat):
    def __init__(self, mean_diff,
                 threshold, direction="unknown", init_stat=0.0):
      #Параметры
        self.mean_hat = 0
        self.std_hat = 1
        self.alpha = 0.05
        self.beta = 0.005
        self.metric = 0
        self.breaks_max = 5
        self.slice_length = 15
        
        # Гиперпараметр Δ
        self.mean_diff = mean_diff 

        self.states = []
        self.breakpoints = []

        self.colors=['black', 'red']
        super(AdjustedCusum, self).__init__(threshold, direction, init_stat)

    def get_stats(self):
        # Оценка среднего и квадрата стандартного отклонения
        try:
            self.mean_hat = self.mean_values_sum / self.mean_weights_sum
            self.var_hat = self.var_values_sum / self.var_weights_sum
        except AttributeError:
            self.mean_hat = 0
            self.var_hat = 1  

    def update_value(self, new_value):

        self.get_stats()
        self.new_value_normalized = (new_value - self.mean_hat) / np.sqrt(self.std_hat)
        
        # Обновление среднего и квадрата стандартного отклонения
        try:
            self.mean_values_sum = (1 - self.alpha) * self.mean_values_sum + new_value
            self.mean_weights_sum = (1 - self.alpha) * self.mean_weights_sum + 1.0
        except AttributeError:
            self.mean_values_sum = new_value
            self.mean_weights_sum = 1.0 
        
        # Задаем новое значение квадратного отклонения
        new_value_var = (self.new_value_normalized - self.mean_hat)**2
        
        try:
            self.var_values_sum = (1 - self.beta) * self.var_values_sum + new_value_var
            self.var_weights_sum = (1 - self.beta) * self.var_weights_sum + 1.0
        except:
            self.var_values_sum = new_value_var
            self.var_weights_sum = 1.0      

    def count(self):
        # Проверяем гипотезу: mean = 0 (при std = 1)
        zeta_k = normal_likelihood(self.new_value_normalized, self.mean_diff, 0., 1)
        self.metric = max(0, self.metric + zeta_k)

        if self.metric > self.threshold:            
            self.states.append(1)
        else:
            self.states.append(0)

        #Если количество разладок превышает максимум сигнал изменением цвета на красный   
        if (np.array(self.states[-self.slice_length:]) == 1).sum() > self.breaks_max:
            self.breakpoints.append('red')
        else:
            self.breakpoints.append('blue')

    def update(self, value):
        zeta_k = normal_likelihood(value, self.mean_diff, 0., 1.)
        self._stat = max(0, self._stat + zeta_k)
        super(AdjustedCusum, self).update()


  #Статистика Ширяева-Робертса для обнаружения разладки в дисперсии временного ряда
class ShiryaevRoberts():
    
    def __init__(self, sigma_diff, threshold=2):
        
        self.mean_hat = 0
        self.std_hat = 1
        
        self.alpha = 0.05
        self.beta = 0.005    
        
        self.metric = 0
        
        # Гиперпараметр Δ
        self.sigma_diff = sigma_diff
        # Верхняя граница для критерия
        self.ceil = 200
        self.threshold = threshold
        self.breaks_max = 3
        self.slice_length = 5
        
        self.states = []
        self.breakpoints = []
        self.colors=['blue', 'red']
        
    def get_stats(self):
        # Оценка среднего и квадрата стандартного отклонения
        try:
            self.mean_hat = self.mean_values_sum / self.mean_weights_sum
            self.var_hat = self.var_values_sum / self.var_weights_sum
        except AttributeError:
            self.mean_hat = 0
            self.var_hat = 1
    
    def update(self, new_value):

        self.get_stats()
        
        # Считаем обновлённое значение среднего и квадрата стандартного отклонения
        self.predicted_diff_value = (new_value - self.mean_hat) ** 2
        self.predicted_diff_mean = self.var_hat
        

        # Обновляем значения среднего и квадрата стандартного отклонения
        try:
            self.mean_values_sum = (1 - self.alpha) * self.mean_values_sum + new_value
            self.mean_weights_sum = (1 - self.alpha) * self.mean_weights_sum + 1.0
        except AttributeError:
            self.mean_values_sum = new_value
            self.mean_weights_sum = 1.0 
        
        # Новое значение квадрата стандартного отклонения
        new_value_var = (new_value - self.mean_hat)**2
        
        try:
            self.var_values_sum = (1 - self.beta) * self.var_values_sum + new_value_var
            self.var_weights_sum = (1 - self.beta) * self.var_weights_sum + 1.0
        except:
            self.var_values_sum = new_value_var
            self.var_weights_sum = 1.0      

    def count(self):
        
        # Проверка гипотезы: среднее разницы между стандартными отклонениями = 0
        adjusted_value = self.predicted_diff_value - self.predicted_diff_mean
        likelihood = np.exp(self.sigma_diff * (adjusted_value - self.sigma_diff / 2.))
        self.metric = min(self.ceil, (1. + self.metric) * likelihood)
        
        if self.metric > self.threshold:            
            self.states.append(1)
        else:
            self.states.append(0)
            
        if (np.array(self.states[-self.slice_length:]) == 1).sum() > self.breaks_max:
            self.breakpoints.append('red')
        else:
            self.breakpoints.append('blue')

class MeanExpNoDataException(Exception):
    pass

# Модификация статистики взвешенного экспоненциального среднего.
class MeanExp(object):
    def __init__(self, new_value_weight, load_function=median):
        self._load_function = load_function
        self._new_value_weight = new_value_weight
        self.load([])

    @property
    def value(self):
        if self._weights_sum <= 1:
            raise MeanExpNoDataException('self._weights_sum <= 1')
        return self._values_sum / self._weights_sum

    def update(self, new_value, **kwargs):
        self._values_sum = (1 - self._new_value_weight) * self._values_sum + new_value
        self._weights_sum = (1 - self._new_value_weight) * self._weights_sum + 1.0

    def load(self, old_values):
        if old_values:
            old_values = [value for ts, value in old_values]
            mean = float(self._load_function(old_values))
            self._weights_sum = min(float(len(old_values)), 1.0 / self._new_value_weight)
            self._values_sum = mean * self._weights_sum
        else:
            self._values_sum = 0.0
            self._weights_sum = 0.0