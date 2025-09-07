

class GradientDescent():
    def get_gradient_descent(self, weights, bias, dW, dB, lr):
        weights = weights - lr * dW
        bias = bias - lr * dB

        return [weights, bias]