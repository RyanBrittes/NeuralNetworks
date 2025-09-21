import numpy as np

class Backward():
    def get_back_reLU(self, dA, Z):
        return dA * (Z > 0)
    
    def get_sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
    
    def get_back_sigmoid(self, dA, Z):
        x = self.get_back_sigmoid(Z)
        return dA * x * (1 - x)
    
    def calc_backward(self, X, Y, parameters, cache):
        n_sample = X.shape(1)
        weights_02 = parameters["weights_02"]

        out_01, out_02 = cache["out_01"], cache["out_02"]
        regression_01, regression_02 = cache["regression_01"], cache["regression_02"]

        d_regression_02 = out_02 - Y
        d_weights_02 = (np.dot(d_regression_02, out_01.T)) / n_sample
        d_bias_02 = (np.sum(d_regression_02, axis=1, keepdims=True)) / n_sample

        d_out_01 = np.dot(weights_02.T, d_regression_02)
        d_regression_01 = self.get_back_reLU(d_out_01, regression_01)
        d_weights_01 = (np.dot(d_regression_01, X.T)) / n_sample
        d_bias_01 = (np.sum(d_regression_01, axis=1, keepdims=True))

        grads = {"d_weights_01": d_weights_01, "d_bias_01": d_bias_01, "d_weights_02": d_weights_02, "d_bias_02": d_bias_02}

        return grads
    