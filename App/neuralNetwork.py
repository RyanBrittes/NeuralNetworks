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
        self.epochs = 1000
        self.lr = 0.01
        self.n_neuron = 4
        self.X = []
        self.Y = []
        
    def train_model(self):
        len_x, len_y = self.X.shape[0], self.Y.shape[0]
        parameters = self.data.initialize_params(len_x, self.n_neuron, len_y)
        
        for epoch in self.epochs:
            out_02, cache = self.forward.calc_forward(self.X, parameters)

            loss = self.loss.calc_binary_cross_entropy(self.Y, out_02, len_y)

            grads = self.backward.calc_backward(self.X, self.Y, parameters, cache)

            parameters = self.optimize.calc_gradient(parameters, grads, self.lr)

            if epoch % 100 == 0:
                print(f"Epoch: {epoch} | Loss: {loss:.4f}")

        return parameters
