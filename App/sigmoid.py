import numpy as np

class Sigmoid():
    def forward(self, X):
        return 1 / (1 + np.exp(-X))

    def backward(self, dA, X):
        z = self.forward(X)
        return dA * z * (1 - z)
    