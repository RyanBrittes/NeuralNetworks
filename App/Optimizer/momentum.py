

class Momentum():
    def get_momentum(self, weights, bias, dW, dB, vW, vB, lr, beta):
        vW = beta * vW + (1 - beta) * dW
        vB = beta * vB + (1 - beta) * dB
        
        weights = weights - lr * dW
        bias = bias - lr * dB

        return [weights, bias, vW, vB]