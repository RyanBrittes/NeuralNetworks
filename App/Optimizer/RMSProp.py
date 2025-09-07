import numpy as np

class RMSProp():
    def get_rms_prop(self, weights, bias, dW, dB, sW, sB, lr, beta, delta):
        sW = beta * sW + (1 - beta) * (dW**2)
        sB = beta * sB + (1 - beta) * (dB**2)
        
        weights = weights - lr * dW / (np.sqrt(sW) + delta)
        bias = bias - lr * dB / (np.sqrt(sB) + delta)

        return [weights, bias, sW, sB]