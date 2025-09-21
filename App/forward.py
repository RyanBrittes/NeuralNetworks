import numpy as np

class Forward():
    def get_reLU(self, Z):
        return np.maximum(0, Z)

    def get_sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def calc_forward(self, X, parameters):
        weights_01, bias_01, weights_02, bias_02 = parameters["weights_01"], parameters["bias_01"], parameters["weights_02"], parameters["bias_02"]

        regression_01 = weights_01 @ X + bias_01
        out_01 = self.get_reLU(regression_01)

        regression_02 = weights_02 @ out_01 + bias_02
        out_02 = self.get_sigmoid(regression_02)

        cache = {"regression_01": regression_01, "out_01": out_01, "regression_02": regression_02, "out_02": out_02}
        
        return out_02, cache
    