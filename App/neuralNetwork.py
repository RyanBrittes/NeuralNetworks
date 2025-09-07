import numpy as np
from ActivationFunction.sigmoid import Sigmoid
from ActivationFunction.reLU import ReLU
from LossFunction.logLoss import LogLoss

class NeuralNetwork():
    def __init__(self):
        self.sigmoid = Sigmoid()
        self.reLU = ReLU()
        self.log_loss = LogLoss()

#Continuar daqui 
