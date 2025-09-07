import numpy as np

class LogLoss():
    def get_log_loss(self, y_true, y_predicted):
        n_sample = y_predicted.shape[1]
        delta = 1e-15
        y_true = np.clip(y_true, delta, 1 - delta)

        loss = -np.mean(y_predicted * np.log(y_true) + (1 - y_predicted) * np.log(1 - y_true))
        dA = -(np.divide(y_predicted, y_true) - np.divide(1 - y_predicted, 1 - y_true)) / n_sample

        return [loss, dA]