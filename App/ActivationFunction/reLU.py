import numpy as np

class ReLU():
    def forward(self, X):
        return np.maximum(0, X)
    
    def backward(self, dA, X):
        dZ = dA.copy()
        dZ[X <= 0] = 0
        return dZ
    