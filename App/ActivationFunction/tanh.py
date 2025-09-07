import numpy as np

class Tanh():
    def forward(self, X):
        return np.tanh(X)
    
    def backward(self, dA, X):
        z = self.forward(X)
        return dA * (1 - z**2)
    