import numpy as np

class MSE():
    def get_MSE(self, y_true, y_predicted):
        n_sample = y_predicted.shape[1]
        loss = np.mean((y_true - y_predicted) ** 2)
        dA = (2/n_sample) * (y_true - y_predicted)

        return [loss, dA]