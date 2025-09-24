import numpy as np
import pandas as pd
from normalizeData import NormalizeData

class LoadData():
    def __init__(self):
        self.normalize = NormalizeData()
        self.__data = pd.read_csv('files/diabetes.csv')
        self.__y_true = self.__data[['Outcome']].values
        self.__x_true = self.__data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']].values

    def get_dataset(self):
        return self.__data

    def get_x_value(self):
        return self.__x_true
    
    def get_y_value(self):
        return self.__y_true
    
    def get_score_Z(self, dataset):
        return self.normalize.calc_score_Z(dataset)
    
    def get_log(self, dataset):
        return self.normalize.calc_log(dataset)
    
    def get_shuffle_separe_train_validation_test(self, rate_test, rate_validation):
        x_values = self.get_score_Z(self.__x_true)
        y_values = self.__y_true

        np.random.seed(42)
        indexShuffled = np.random.permutation(len(y_values))

        y_shuffled = y_values[indexShuffled]
        x_shuffled = x_values[indexShuffled]
        
        rate_train = 1 - rate_test - rate_validation

        len_sample = len(y_values)
        len_train = np.floor(len_sample * rate_train).astype(int)
        len_validation = np.round(len_sample * rate_validation).astype(int)

        x_train = x_shuffled[0:len_train].T
        y_train = y_shuffled[0:len_train].T
        x_validation = x_shuffled[len_train:(len_train + len_validation)].T
        y_validation = y_shuffled[len_train:(len_train + len_validation)].T
        x_test = x_shuffled[(len_train + len_validation):len_sample].T
        y_test = y_shuffled[(len_train + len_validation):len_sample].T
        
        return [x_train, y_train, x_validation, y_validation, x_test, y_test, len_sample]

    def initialize_params(self, len_x, n_neuron, len_y):
        np.random.seed(1)

        weights_01 = np.random.randn(n_neuron, len_x) * np.sqrt(2.0 / len_x)
        bias_01 = np.zeros((n_neuron, 1))

        weights_02 = np.random.randn(len_y, n_neuron) * np.sqrt(2.0 / n_neuron)
        bias_02 = np.zeros((len_y, 1))

        return {"weights_01": weights_01, "bias_01": bias_01, "weights_02": weights_02, "bias_02": bias_02}
    