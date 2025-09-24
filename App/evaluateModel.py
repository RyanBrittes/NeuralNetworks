import numpy as np
from neuralNetwork import NeuralNetwork
from forward import Forward

class EvaluateModel():
    def __init__(self):
        self.trained_params = NeuralNetwork()
        self.forward = Forward()

    def get_predict(self):
        parameters, _, x_test, y_test = self.trained_params.train_model()

        out_02, _ = self.forward.calc_forward(x_test, parameters)
        predict = (out_02 > 0.5).astype(int)

        accuracy = np.mean(predict == y_test)

        return [predict, accuracy]
