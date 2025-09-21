import numpy as np
from loadData import LoadData
from forward import Forward
from backward import Backward
from lossFunction import LossFunction
from gradientDescent import GradientDescent

class NeuralNetwork():
    def __init__(self):
        self.forward = Forward()
        self.backward = Backward()
        self.data = LoadData()
        self.loss = LossFunction()
        self.optimize = GradientDescent()
        self.n_neuron = 4
        self.rate_test = 0.2
        self.rate_validation = 0
        self.shuffled_data = self.data.get_shuffle_separe_train_validation_test(self.rate_test, self.rate_validation)
        self.x_train = self.shuffled_data[0]
        self.y_train = self.shuffled_data[1]
        self.x_validation = self.shuffled_data[2]
        self.y_validation = self.shuffled_data[3]
        self.x_test = self.shuffled_data[4]
        self.y_test = self.shuffled_data[5]
        self.len_sample = self.shuffled_data[6]
        self.lr = 0.01
        self.epochs = 8000
        self.losses = []
        self.batch_size = 50
        
    def train_model(self):
        len_x, len_y = self.x_train.shape[0], self.y_train.shape[0]
        parameters = self.data.initialize_params(len_x, self.n_neuron, len_y)
        
        for epoch in self.epochs:
            out_02, cache = self.forward.calc_forward(self.x_train, parameters)

            loss = self.loss.calc_binary_cross_entropy(self.y_train, out_02, len_y)
            self.losses.append(loss)

            grads = self.backward.calc_backward(self.x_train, self.y_train, parameters, cache)

            parameters = self.optimize.calc_gradient(parameters, grads, self.lr)

            if epoch % 100 == 0:
                print(f"Epoch: {epoch} | Loss: {loss:.4f}")

        return [parameters, self.losses, self.x_test, self.y_test]
