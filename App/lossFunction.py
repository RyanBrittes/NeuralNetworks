import numpy as np

class LossFunction():
    def calc_log_loss(self, y_pred, y_true):
        return np.mean(-y_true * np.log(y_pred + 1e-15) - (1 - y_true) * np.log(1 - y_pred + 1e-15))
    
    def calc_mse_loss(self, y_pred, y_true):
        return np.mean((y_true - y_pred) ** 2)

    def calc_simple_loss(self, y_pred, y_true):
        return (y_true - y_pred)
    
    def calc_binary_cross_entropy(self, y_true, y_pred, n_sample):
        loss = - (np.sum(y_true * np.log(y_pred + 1e-16) + (1 - y_true) * np.log(1 - y_pred + 1e-16))) / n_sample

        return np.squeeze(loss)